#include "kimm_polaris3d/polaris3d_hqp.h"

using namespace pinocchio;
using namespace Eigen;
using namespace std;
using namespace kimmhqp;
using namespace kimmhqp::trajectory;
using namespace kimmhqp::math;
using namespace kimmhqp::tasks;
using namespace kimmhqp::solver;
using namespace kimmhqp::robot;
using namespace kimmhqp::contacts;

namespace RobotController{
    FrankaWrapper::FrankaWrapper(const std::string & robot_node, const bool & issimulation, ros::NodeHandle & node)
    : robot_node_(robot_node), issimulation_(issimulation), n_node_(node)
    {
        time_ = 0.;        
        node_index_ = 0;
        cnt_ = 0;

        mode_change_ = false;
        ctrl_mode_ = 0;       
    }

    void FrankaWrapper::initialize(){
        // Robot for pinocchio
        string model_path, urdf_name;
        n_node_.getParam("/" + robot_node_ +"/robot_urdf_path", model_path);
        n_node_.getParam("/" + robot_node_ +"/robot_urdf", urdf_name);        //"panda_arm_hand_l.urdf"

        vector<string> package_dirs;
        package_dirs.push_back(model_path);
        string urdfFileName = package_dirs[0] + urdf_name;
        robot_ = std::make_shared<RobotWrapper>(urdfFileName, package_dirs, false, false); //first false : w/o mobile, true : w/ mobile
        model_ = robot_->model();
        
        //nq_/nv_/na_ is # of joint w.r.t pinocchio model ("panda_arm_hand_l.urdf"), so there is no gripper joints
        nq_ = robot_->nq(); //7 : franka (7) 
        nv_ = robot_->nv(); //7
        na_ = robot_->na(); //7 

        // State (for pinocchio)
        state_.q_.setZero(nq_);
        state_.v_.setZero(nv_);
        state_.dv_.setZero(nv_);
        state_.torque_.setZero(na_);
        state_.tau_.setZero(na_);

        // tsid
        tsid_ = std::make_shared<InverseDynamicsFormulationAccForce>("tsid", *robot_);
        tsid_->computeProblemData(time_, state_.q_, state_.v_);
        data_ = tsid_->data();

        // tasks
        postureTask_ = std::make_shared<TaskJointPosture>("task-posture", *robot_);
        VectorXd posture_gain(na_);
        if (!issimulation_) //for real        	            
            // posture_gain << 100., 100., 100., 200., 200., 200., 200.;
            posture_gain << 300., 300., 300., 600., 600., 600., 1200.;
        else // for simulation        	
            // posture_gain << 40000., 40000., 40000., 40000., 40000., 40000., 40000.;            
            posture_gain << 80000., 80000., 80000., 80000., 80000., 80000., 80000.;            

        postureTask_->Kp(posture_gain);
        postureTask_->Kd(2.0*postureTask_->Kp().cwiseSqrt());
        
        //////////////////// EE offset ////////////////////////////////////////                
        //When offset is applied, the control characteristics seems to be changed (gain tuning is needed)
        //This offset is applied to both reference (in this code) and feedback (in task_se3_equality.cpp)
        //robotiq
        joint7_to_finger_ = 0.247; //0.247 (z-axis) = 0.222(distance from link7 to left_inner_finger) + 0.025(finger center length)                 
        this->eeoffset_update(); //pair with home position            
        ///////////////////////////////////////////////////////////////////////        

        VectorXd ee_gain(6);        
        if (!issimulation_) { //for real                
            ee_gain << 100., 100., 100., 400., 400., 600.;                
        }
        else { //for simulation
            // ee_gain << 500., 500., 500., 800., 800., 1000.;
            ee_gain << 1000., 1000., 1000., 2000., 2000., 2000.;
        }  

        eeTask_ = std::make_shared<TaskSE3Equality>("task-se3", *robot_, "panda_joint7", ee_offset_); //here, ee_offset_ is applied to current pos value
        eeTask_->Kp(ee_gain*Vector::Ones(6));
        eeTask_->Kd(2.0*eeTask_->Kp().cwiseSqrt());        
        // eeTask_->Kd(0.5*eeTask_->Kp().cwiseSqrt());
      
        torqueBoundsTask_ = std::make_shared<TaskJointBounds>("task-torque-bounds", *robot_);
        Vector dq_max = 500000.0*Vector::Ones(na_);
        dq_max(0) = 500.; //? 
        dq_max(1) = 500.; //?
        Vector dq_min = -dq_max;
        torqueBoundsTask_->setJointBounds(dq_min, dq_max);        

        // trajecotries
        sampleEE_.resize(12, 6); //12=3(translation)+9(rotation matrix), 6=3(translation)+3(rotation)
        samplePosture_.resize(na_); //na_=7 franka 7

        trajPosture_Cubic_ = std::make_shared<TrajectoryEuclidianCubic>("traj_posture");
        trajPosture_Constant_ = std::make_shared<TrajectoryEuclidianConstant>("traj_posture_constant");
        trajPosture_Timeopt_ = std::make_shared<TrajectoryEuclidianTimeopt>("traj_posture_timeopt");

        trajEE_Cubic_ = std::make_shared<TrajectorySE3Cubic>("traj_ee");
        trajEE_Constant_ = std::make_shared<TrajectorySE3Constant>("traj_ee_constant");
        Vector3d Maxvel_ee = Vector3d::Ones()*0.2;
        Vector3d Maxacc_ee = Vector3d::Ones()*0.2;
        trajEE_Timeopt_ = std::make_shared<TrajectorySE3Timeopt>("traj_ee_timeopt", Maxvel_ee, Maxacc_ee);

        // solver
        solver_ = SolverHQPFactory::createNewSolver(SOLVER_HQP_QPOASES, "qpoases");

        // service
        reset_control_ = true;         

        //inertia shaping
        Me_inv_.resize(6,6);
        Me_inv_.setIdentity();                
        eeTask_->setDesiredinertia(MatrixXd::Identity(6,6)); 
    }    

    void FrankaWrapper::eeoffset_update(){         
        ee_offset_ = Vector3d(0.0, 0.0, joint7_to_finger_); //w.r.t joint7

        T_offset_.setIdentity();
        T_offset_.translation(ee_offset_);                

        Adj_mat_.resize(6,6);
        Adj_mat_.setIdentity();
        Adj_mat_.topRightCorner(3,3) = -1 * skew_matrix(ee_offset_); //due to "A cross B = -B cross A"
    }

    void FrankaWrapper::franka_update(const sensor_msgs::JointState& msg){ //for simulation (mujoco)
        // mujoco callback msg
        // msg.position : 7(joint) + 2(gripper)
        // msg.velocity : 7(joint) + 2(gripper)

        assert(issimulation_);
        for (int i=0; i< nq_; i++){
            state_.q_(i) = msg.position[i];
            state_.v_(i) = msg.velocity[i];
        }
    }
    void FrankaWrapper::franka_update(const Vector7d& q, const Vector7d& qdot){ //for experiment
        assert(!issimulation_);
        state_.q_.tail(nq_) = q;
        state_.v_.tail(nq_) = qdot;
    }        

    void FrankaWrapper::franka_update(const Vector7d& q, const Vector7d& qdot, const Vector7d& tau){ //for experiment, use pinocchio::aba
        assert(!issimulation_);
        state_.q_.tail(nq_) = q;
        state_.v_.tail(nv_) = qdot;

        state_.tau_.tail(na_) = tau;
    }         

    void FrankaWrapper::Fext_update(const Vector6d& Fext){ //for simulation & experiment
        Fext_ = Fext;

        // cout << Fext_.transpose() << endl;
    }

    void FrankaWrapper::ctrl_update(const int& msg){
        ctrl_mode_ = msg;
        ROS_INFO("[ctrltypeCallback] %d", ctrl_mode_);
        mode_change_ = true;
    }

    void FrankaWrapper::compute(const double& time){
        time_ = time;

        robot_->computeAllTerms(data_, state_.q_, state_.v_);
        // robot_->computeAllTerms_ABA(data_, state_.q_, state_.v_, state_.tau_); //to try to use data.ddq (only computed from ABA) However,the ddq value with ABA is not reasonalbe.        

        if (ctrl_mode_ == 0){ //g // gravity mode
            state_.torque_.setZero();
        }
        if (ctrl_mode_ == 1){ //h //init position
            if (mode_change_){                
                //remove
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");
                
                //add
                tsid_->addMotionTask(*postureTask_, 1e-2, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);                

                //posture
                q_ref_.setZero(7);
                q_ref_(0) =  0.0;
                q_ref_(1) =  0.0;
                q_ref_(3) = -M_PI / 2.0;
                q_ref_(5) =  M_PI / 2.0;
                // q_ref_(6) = -M_PI / 2.0;
                q_ref_(6) =  0.0;

                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(4.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(q_ref_);

                reset_control_ = false;
                mode_change_ = false;             
            }            

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            //in here, task.compute is performed right after the reference is set.
            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_); 

            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        }  

        if (ctrl_mode_ == 11){ //q //init position w.r.t ereon
            if (mode_change_){                
                //remove
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-2, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                
                //posture
                q_ref_.setZero(7);
                q_ref_(0) =  5.0 * M_PI / 180.0;
                q_ref_(1) =  45.0 * M_PI / 180.0;
                q_ref_(2) = -11.0 * M_PI / 180.0;
                q_ref_(3) = -118.0 * M_PI / 180.0;
                q_ref_(4) =  80.0  * M_PI / 180.0;
                q_ref_(5) =  110.0  * M_PI / 180.0;
                q_ref_(6) = -74.0  * M_PI / 180.0;

                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(4.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(q_ref_);                

                reset_control_ = false;
                mode_change_ = false;             
            }            

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            //in here, task.compute is performed right after the reference is set.
            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_); 

            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        }   

        if (ctrl_mode_ == 12){ //w //transition pos to place bottle
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current4 joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(4.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(4.0);

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
                trajEE_Cubic_->setInitSample(H_ee_ref_);                       
                
                pinocchio::SE3 T_aruco_offset;                
                T_aruco_offset.setIdentity();
                T_aruco_offset.translation(Vector3d(0.0, -0.17, -0.15));
                trajEE_Cubic_->setGoalSample(T_aruco_ * T_aruco_offset);

                reset_control_ = false;
                mode_change_ = false;                
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));   
        }             

        if (ctrl_mode_ == 13){ //e //go to ereon plate to place bottle
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current4 joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(4.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(4.0);

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
                trajEE_Cubic_->setInitSample(H_ee_ref_);                       
                
                pinocchio::SE3 T_aruco_offset;                
                T_aruco_offset.setIdentity();
                T_aruco_offset.translation(Vector3d(0.0, -0.12, 0.05));
                trajEE_Cubic_->setGoalSample(T_aruco_ * T_aruco_offset);

                reset_control_ = false;
                mode_change_ = false;                
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));   
        } 
        
        if (ctrl_mode_ == 21){ //a //init position w.r.t. bottle
            if (mode_change_){                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                tsid_->addMotionTask(*postureTask_, 1e-2, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);

                q_ref_.setZero(7);
                q_ref_(0) = -13.0 * M_PI / 180.0;
                q_ref_(1) =  30.0 * M_PI / 180.0;
                q_ref_(2) =  11.0 * M_PI / 180.0;
                q_ref_(3) = -123.0 * M_PI / 180.0;
                q_ref_(4) = -80.0  * M_PI / 180.0;
                q_ref_(5) =  113.0  * M_PI / 180.0;
                q_ref_(6) = -118.0  * M_PI / 180.0;

                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(4.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(q_ref_);

                reset_control_ = false;
                mode_change_ = false;             
            }            

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            //in here, task.compute is performed right after the reference is set.
            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_); 

            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        } 

        if (ctrl_mode_ == 22){ //s //transition pos to pick bottle
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current4 joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(4.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(4.0);

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
                trajEE_Cubic_->setInitSample(H_ee_ref_);                       
                
                pinocchio::SE3 T_aruco_offset;                
                T_aruco_offset.setIdentity();
                T_aruco_offset.translation(Vector3d(-0.02, -0.13, -0.1));
                trajEE_Cubic_->setGoalSample(T_aruco_ * T_aruco_offset);

                reset_control_ = false;
                mode_change_ = false;                
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));   
        }             

        if (ctrl_mode_ == 23){ //d //bottle pick pos
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current4 joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(4.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(4.0);

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
                trajEE_Cubic_->setInitSample(H_ee_ref_);                       
                
                pinocchio::SE3 T_aruco_offset;                
                T_aruco_offset.setIdentity();
                T_aruco_offset.translation(Vector3d(-0.02, -0.10, 0.1));
                trajEE_Cubic_->setGoalSample(T_aruco_ * T_aruco_offset);

                reset_control_ = false;
                mode_change_ = false;                
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));   
        } 
        
        // --------------------------------------------------------------------------------------------------------//
        // pick bottle with AI ------------------------------------------------------------------------------------//
        // --------------------------------------------------------------------------------------------------------//
        
        // if (ctrl_mode_ == 21){ //a //init position w.r.t. bottle
        //     if (mode_change_){
        //         //remove                
        //         tsid_->removeTask("task-se3");
        //         tsid_->removeTask("task-posture");
        //         tsid_->removeTask("task-torque-bounds");

        //         //add
        //         tsid_->addMotionTask(*postureTask_, 1e-16, 1);
        //         tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
        //         tsid_->addMotionTask(*eeTask_, 1.0, 0);

        //         //posture (try to maintain current4 joint configuration)
        //         trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
        //         trajPosture_Cubic_->setDuration(4.0);
        //         trajPosture_Cubic_->setStartTime(time_);
        //         trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

        //         //ee
        //         trajEE_Cubic_->setStartTime(time_);
        //         trajEE_Cubic_->setDuration(4.0);

        //         H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
        //         trajEE_Cubic_->setInitSample(H_ee_ref_);                       
                
        //         pinocchio::SE3 T_yolo_offset;                
        //         T_yolo_offset.setIdentity();
        //         // T_yolo_offset.translation(Vector3d(0.1, 0.1, 0.0));
        //         trajEE_Cubic_->setGoalSample(T_yolo_ * T_yolo_offset);

        //         reset_control_ = false;
        //         mode_change_ = false;                
        //     }

        //     trajPosture_Cubic_->setCurrentTime(time_);
        //     samplePosture_ = trajPosture_Cubic_->computeNext();
        //     postureTask_->setReference(samplePosture_);

        //     trajEE_Cubic_->setCurrentTime(time_);
        //     sampleEE_ = trajEE_Cubic_->computeNext();
        //     eeTask_->setReference(sampleEE_);

        //     const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
        //     state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));   
        // }   

        // --------------------------------------------------------------------------------------------------------//
        // impedance control ------------------------------------------------------------------------------------//
        // --------------------------------------------------------------------------------------------------------//
     
        // if (ctrl_mode_ == 22){ //v //impedance control
        //     if (mode_change_){
        //         //remove                
        //         tsid_->removeTask("task-se3");
        //         tsid_->removeTask("task-posture");
        //         tsid_->removeTask("task-torque-bounds");

        //         //add
        //         tsid_->addMotionTask(*postureTask_, 1e-6, 1);
        //         tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
        //         tsid_->addMotionTask(*eeTask_, 1.0, 0);

        //         //posture
        //         trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
        //         trajPosture_Cubic_->setDuration(2.0);
        //         trajPosture_Cubic_->setStartTime(time_);
        //         trajPosture_Cubic_->setGoalSample(q_ref_);

        //         //ee
        //         trajEE_Cubic_->setStartTime(time_);
        //         trajEE_Cubic_->setDuration(2.0);
        //         H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                

        //         trajEE_Cubic_->setInitSample(H_ee_ref_);
        //         trajEE_Cubic_->setGoalSample(H_ee_ref_);
               
        //         reset_control_ = false;
        //         mode_change_ = false;                                                
        //     }
            
        //     trajPosture_Cubic_->setCurrentTime(time_);
        //     samplePosture_ = trajPosture_Cubic_->computeNext();
        //     postureTask_->setReference(samplePosture_);                        
            
        //     if (1) { 
        //         //to make K(x-xd)=0, put xd=x 
        //         H_ee_ref_.translation() = (robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_).translation();
                
        //         //inertia shaping    
        //         // Me_inv_ = 
        //         eeTask_->setDesiredinertia(Me_inv_);
        //     }
        //     else {   
        //         //to make K(x-xd)=0, put K=0 
        //         Vector6d a;
        //         a = eeTask_->Kp();
        //         a.head(3) << 0.0, 0.0, 0.0;
        //         eeTask_->Kp(a); //after this task, Kp should be returned to original gain
        //     }                        

        //     SE3ToVector(H_ee_ref_, sampleEE_.pos);

        //     eeTask_->setReference(sampleEE_);
        //     //////////////////

        //     const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
        //     state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        // }            
        
        if (ctrl_mode_ == 31){ //i //move ee +0.1z
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
                trajEE_Cubic_->setInitSample(H_ee_ref_);
                H_ee_ref_.translation()(2) += 0.1;                
                trajEE_Cubic_->setGoalSample(H_ee_ref_);

                reset_control_ = false;
                mode_change_ = false;                
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));   
        }     

        if (ctrl_mode_ == 32){ //k //move ee -0.1z
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
                trajEE_Cubic_->setInitSample(H_ee_ref_);
                H_ee_ref_.translation()(2) -= 0.1;                
                trajEE_Cubic_->setGoalSample(H_ee_ref_);

                reset_control_ = false;
                mode_change_ = false;                
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));   
        }  

        if (ctrl_mode_ == 33){ //l //move ee +0.1x
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
                trajEE_Cubic_->setInitSample(H_ee_ref_);
                H_ee_ref_.translation()(0) += 0.1;                
                trajEE_Cubic_->setGoalSample(H_ee_ref_);

                reset_control_ = false;
                mode_change_ = false;                
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));            
        }

        if (ctrl_mode_ == 34){ //j //move ee -0.1x
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
                trajEE_Cubic_->setInitSample(H_ee_ref_);
                H_ee_ref_.translation()(0) -= 0.1;                
                trajEE_Cubic_->setGoalSample(H_ee_ref_);

                reset_control_ = false;
                mode_change_ = false;                
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));            
        }

        if (ctrl_mode_ == 35){ //u //move ee -0.1y
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
                trajEE_Cubic_->setInitSample(H_ee_ref_);
                H_ee_ref_.translation()(1) -= 0.1;                
                trajEE_Cubic_->setGoalSample(H_ee_ref_);

                reset_control_ = false;
                mode_change_ = false;                
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));            
        }

        if (ctrl_mode_ == 36){ //o //move ee 0.1y
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                                                
                trajEE_Cubic_->setInitSample(H_ee_ref_);
                H_ee_ref_.translation()(1) += 0.1;                
                trajEE_Cubic_->setGoalSample(H_ee_ref_);

                reset_control_ = false;
                mode_change_ = false;                
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));            
        }          

        if (ctrl_mode_ == 99){ //p //print current ee state
            if (mode_change_){
                //remove               
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-6, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //traj
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);
                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;                
                trajEE_Cubic_->setInitSample(H_ee_ref_);
                trajEE_Cubic_->setGoalSample(H_ee_ref_);

                // cout << H_ee_ref_ << endl;

                cout << robot_->position(data_, robot_->model().getJointId("panda_joint7"))  << endl;
                cout << robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_ << endl;

                reset_control_ = false;
                mode_change_ = false;
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        }
    }

    void FrankaWrapper::franka_output(VectorXd & qacc) { //from here to main code
        qacc = state_.torque_.tail(na_);
    }    

    void FrankaWrapper::com(Eigen::Vector3d & com){
        //API:Vector of subtree center of mass positions expressed in the root joint of the subtree. 
        //API:In other words, com[j] is the CoM position of the subtree supported by joint j and expressed in the joint frame .         
        //API:The element com[0] corresponds to the center of mass position of the whole model and expressed in the global frame.
        com = robot_->com(data_);
    }

    void FrankaWrapper::position(pinocchio::SE3 & oMi){
        //API:Vector of absolute joint placements (wrt the world).
        oMi = robot_->position(data_, robot_->model().getJointId("panda_joint7"));
    }

    void FrankaWrapper::position_offset(pinocchio::SE3 & oMi){
        //API:Vector of absolute joint placements (wrt the world).
        oMi = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;
    }

    void FrankaWrapper::velocity(pinocchio::Motion & vel){
        //API:Vector of joint velocities expressed at the centers of the joints.
        vel = robot_->velocity(data_, robot_->model().getJointId("panda_joint7"));

        // NOTES ////////////////////////////////////////////////////////////////////////////
        // data.v = f.placement.actInv(data.v[f.parent]), both are in LOCAL coordinate
        // T_offset.act(v_frame) is WRONG method, It means that applying offset w.r.t. global coord. to EE frame.                
        //////////////////////////////////////////////////////////////////////////////////////

        // code comparison ///////////////////////////
        // w/o offset, three resaults are identical 
        // w/ offset, last two resaults are identical
        //////////////////////////////////////////////                                
        // cout << "data.v" << endl;
        // cout <<  robot_->velocity(data_, robot_->model().getJointId("panda_joint7")) << endl;        

        // cout << "data.v with Adj_mat_" << endl;
        // cout << vel.linear() + Adj_mat_.topRightCorner(3,3) * vel.angular() << endl;
        // cout << vel.angular() << endl;        
    }

    void FrankaWrapper::velocity_origin(pinocchio::Motion & vel){
        //API:Vector of joint velocities expressed at the origin. (data.ov)
        //Same with "vel = m_wMl.act(v_frame);"
        vel = robot_->velocity_origin(data_, robot_->model().getJointId("panda_joint7"));
    }

    void FrankaWrapper::acceleration(pinocchio::Motion & accel){        
        //API:Vector of joint accelerations expressed at the centers of the joints frames. (data.a)
        accel = robot_->acceleration(data_, robot_->model().getJointId("panda_joint7"));                                        
    }

    void FrankaWrapper::acceleration_origin(pinocchio::Motion & accel){
        //API:Vector of joint accelerations expressed at the origin of the world. (data.oa)        
        //It is not available!!!!!!!!!!!!!!! (always zero), so acceleratioon_global is used.
        accel = robot_->acceleration_origin(data_, robot_->model().getJointId("panda_joint7"));
    }

    void FrankaWrapper::acceleration_origin2(pinocchio::Motion & accel){
        //It is defined becuase data.oa is not available (output always zero) 
        SE3 m_wMl;
        Motion a_frame;        
        robot_->framePosition(data_, robot_->model().getFrameId("panda_joint7"), m_wMl);       //data.oMi     
        robot_->frameAcceleration(data_, robot_->model().getFrameId("panda_joint7"), a_frame); //data.a         
        accel = m_wMl.act(a_frame);
    }

    void FrankaWrapper::force(pinocchio::Force & force){
        //API:Vector of body forces expressed in the local frame of the joint. 
        //API:For each body, the force represents the sum of all external forces acting on the body.
        force = robot_->force(data_, robot_->model().getJointId("panda_joint7"));
    }

    void FrankaWrapper::force_origin(pinocchio::Force & force){
        //API:Vector of body forces expressed in the world frame. 
        //API:For each body, the force represents the sum of all external forces acting on the body.        
        //It is not available!!!!!!!!!!!!!!! (always zero), so acceleratioon_global is used.
        force = robot_->force_origin(data_, robot_->model().getJointId("panda_joint7"));
    }

    void FrankaWrapper::force_origin2(pinocchio::Force & force){
        //It is defined becuase data.of is not available (output always zero)         
        SE3 m_wMl;
        Force f_frame;
        robot_->framePosition(data_, robot_->model().getFrameId("panda_joint7"), m_wMl);
        robot_->frameForce(data_, robot_->model().getFrameId("panda_joint7"), f_frame);
        force = m_wMl.act(f_frame);
    }

    void FrankaWrapper::tau(VectorXd & tau_vec){
        //API:Vector of joint torques (dim model.nv).
        //It is not available (output always zero)
        tau_vec = robot_->jointTorques(data_).tail(na_);
    }

    void FrankaWrapper::ddq(VectorXd & ddq_vec){
        //API:The joint accelerations computed from ABA.
        //It is not available (output always zero), will be available with ABA method
        //even though with ABA method, the value is not reasonable (behave like torque, not ddq)
        ddq_vec = robot_->jointAcceleration(data_).tail(na_);
    }

    void FrankaWrapper::mass(MatrixXd & mass_mat){
        mass_mat = robot_->mass(data_).bottomRightCorner(na_, na_);
    }

    void FrankaWrapper::nle(VectorXd & nle_vec){
        nle_vec = robot_->nonLinearEffects(data_).tail(na_);
    }

    void FrankaWrapper::g(VectorXd & g_vec){
        //API:Vector of generalized gravity (dim model.nv).
        g_vec = data_.g.tail(na_);
    }

    void FrankaWrapper::g_joint7(VectorXd & g_vec){
        Vector3d g_global;
        // g_global << 0.0, 0.0, -9.81;
        g_global << 0.0, 0.0, 9.81;
        
        SE3 m_wMl;                
        robot_->framePosition(data_, robot_->model().getFrameId("panda_joint7"), m_wMl);
        m_wMl.translation() << 0.0, 0.0, 0.0; //transform only with rotation
        g_vec = m_wMl.actInv(g_global);
    }

    void FrankaWrapper::g_local_offset(VectorXd & g_vec){
        Vector3d g_global;
        // g_global << 0.0, 0.0, -9.81;
        g_global << 0.0, 0.0, 9.81;
        
        SE3 m_wMl;                
        m_wMl = robot_->position(data_, robot_->model().getJointId("panda_joint7")) * T_offset_;
        m_wMl.translation() << 0.0, 0.0, 0.0; //transform only with rotation
        
        g_vec = m_wMl.actInv(g_global);
    }

    void FrankaWrapper::JWorld(MatrixXd & Jo){
        Data::Matrix6x Jo2;        
        Jo2.resize(6, robot_->nv());
        robot_->jacobianWorld(data_, robot_->model().getJointId("panda_joint7"), Jo2);
        Jo = Jo2.bottomRightCorner(6, 7);
    }

    void FrankaWrapper::JLocal(MatrixXd & Jo){
        Data::Matrix6x Jo2;        
        Jo2.resize(6, robot_->nv());
        robot_->frameJacobianLocal(data_, robot_->model().getFrameId("panda_joint7"), Jo2);        
        Jo = Jo2.bottomRightCorner(6, 7);
    }

    void FrankaWrapper::JLocal_offset(MatrixXd & Jo){
        Data::Matrix6x Jo2;        
        Jo2.resize(6, robot_->nv());
        robot_->frameJacobianLocal(data_, robot_->model().getFrameId("panda_joint7"), Jo2);        
        Jo = Jo2.bottomRightCorner(6, 7);
        Jo = Adj_mat_ * Jo;
    }

    void FrankaWrapper::dJLocal(MatrixXd & dJo){
        Data::Matrix6x dJo2;        
        dJo2.resize(6, robot_->nv());
        robot_->frameJacobianTimeVariationLocal(data_, robot_->model().getFrameId("panda_joint7"), dJo2);        
        dJo = dJo2.bottomRightCorner(6, 7);
    }

    void FrankaWrapper::dJLocal_offset(MatrixXd & dJo){
        Data::Matrix6x dJo2;        
        dJo2.resize(6, robot_->nv());
        robot_->frameJacobianTimeVariationLocal(data_, robot_->model().getFrameId("panda_joint7"), dJo2);        
        dJo = dJo2.bottomRightCorner(6, 7);
        dJo = Adj_mat_ * dJo;
    }

    void FrankaWrapper::ee_state(Vector3d & pos, Eigen::Quaterniond & quat){
        for (int i=0; i<3; i++)
            pos(i) = robot_->position(data_, robot_->model().getJointId("panda_joint7")).translation()(i);

        Quaternion<double> q(robot_->position(data_, robot_->model().getJointId("panda_joint7")).rotation());
        quat = q;
    }

    void FrankaWrapper::rotx(double & angle, Eigen::Matrix3d & rot){
        rot.row(0) << 1,           0,           0;
        rot.row(1) << 0,           cos(angle), -sin(angle);
        rot.row(2) << 0,           sin(angle),  cos(angle);
    }

    void FrankaWrapper::roty(double & angle, Eigen::Matrix3d & rot){
        rot.row(0) <<  cos(angle),           0,            sin(angle);
        rot.row(1) <<  0,                    1,            0 ;
        rot.row(2) << -sin(angle),           0,            cos(angle);
    }

    void FrankaWrapper::rotz(double & angle, Eigen::Matrix3d & rot){
        rot.row(0) << cos(angle), -sin(angle),  0;
        rot.row(1) << sin(angle),  cos(angle),  0;
        rot.row(2) << 0,                    0,  1;
    }    
    
    MatrixXd FrankaWrapper::skew_matrix(const VectorXd& vec){
        double v1 = vec(0);
        double v2 = vec(1);
        double v3 = vec(2);

        Eigen::Matrix3d CM;
        CM <<     0, -1*v3,    v2,
                 v3,     0, -1*v1,
              -1*v2,    v1,     0;

        return CM;
    }

    pinocchio::SE3 FrankaWrapper::vel_to_SE3(VectorXd vel, double dt){        
        Quaterniond angvel_quat(1, vel(3) * dt * 0.5, vel(4) * dt * 0.5, vel(5) * dt * 0.5);
        Matrix3d Rotm = angvel_quat.normalized().toRotationMatrix();        
        
        pinocchio::SE3 T;
        T.translation() = vel.head(3) * dt;
        T.rotation() = Rotm;

        // cout << T << endl;

        return T;
    }

    void FrankaWrapper::get_dt(double dt){
        dt_ = dt;
    }

    double FrankaWrapper::noise_elimination(double x, double limit) {
        double y;
        if (abs(x) > limit) y = x;
        else y = 0.0;
        
        return y;
    }

    void FrankaWrapper::get_aruco_marker(geometry_msgs::Pose pos) { 
        T_aruco_ = SE3(Eigen::Quaterniond(pos.orientation.w, pos.orientation.x, pos.orientation.y, pos.orientation.z).toRotationMatrix(),
                      Eigen::Vector3d(pos.position.x, pos.position.y, pos.position.z)); 
        cout << T_aruco_ << endl;
    }

    void FrankaWrapper::get_yolo_marker(geometry_msgs::Pose pos) { 
        T_yolo_ = SE3(Eigen::Quaterniond(pos.orientation.w, pos.orientation.x, pos.orientation.y, pos.orientation.z).toRotationMatrix(),
                      Eigen::Vector3d(pos.position.x, pos.position.y, pos.position.z)); 
        cout << T_yolo_ << endl;
    }
}// namespace
