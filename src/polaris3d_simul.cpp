#include "kimm_polaris3d/polaris3d_simul.h"

using namespace std;
using namespace pinocchio;
using namespace Eigen;
using namespace RobotController;

int main(int argc, char **argv)
{   
    //Ros setting
    ros::init(argc, argv, "kimm_phri_panda");
    ros::NodeHandle n_node;
    
    dt = 0.001;
    time_ = 0.0;
    ros::Rate loop_rate(1.0/dt);

    /////////////// Robot Wrapper ///////////////
    n_node.getParam("/robot_group", group_name);    
    ctrl_ = new RobotController::FrankaWrapper(group_name, true, n_node);
    ctrl_->initialize();
    ctrl_->get_dt(dt);
    
    /////////////// mujoco sub : from mujoco to here ///////////////    
    ros::Subscriber jointState = n_node.subscribe("mujoco_ros/mujoco_ros_interface/joint_states", 5, &JointStateCallback, ros::TransportHints().tcpNoDelay(true));    
    ros::Subscriber mujoco_command_sub = n_node.subscribe("mujoco_ros/mujoco_ros_interface/sim_command_sim2con", 5, &simCommandCallback, ros::TransportHints().tcpNoDelay(true));
    ros::Subscriber mujoco_time_sub = n_node.subscribe("mujoco_ros/mujoco_ros_interface/sim_time", 1, &simTimeCallback, ros::TransportHints().tcpNoDelay(true));
    ros::Subscriber ctrl_type_sub = n_node.subscribe("mujoco_ros/mujoco_ros_interface/ctrl_type", 1, &ctrltypeCallback, ros::TransportHints().tcpNoDelay(true));
    ros::Subscriber aruco_sub = n_node.subscribe("kimm_aruco_publisher/pose", 1, &ArucoCallback, ros::TransportHints().tcpNoDelay(true)); 
    // ros::Subscriber yolo_sub = n_node.subscribe("darknet_ros_3d/bounding_boxes", 1, &YoloCallback, ros::TransportHints().tcpNoDelay(true)); 
    ros::Subscriber yolo_sub = n_node.subscribe("darknet_ros_3d/markers", 1, &YoloCallback, ros::TransportHints().tcpNoDelay(true)); 

    /////////////// mujoco pub : from here to mujoco ///////////////    
    mujoco_command_pub_ = n_node.advertise<std_msgs::String>("mujoco_ros/mujoco_ros_interface/sim_command_con2sim", 5);
    robot_command_pub_ = n_node.advertise<mujoco_ros_msgs::JointSet>("mujoco_ros/mujoco_ros_interface/joint_set", 5);
    mujoco_run_pub_ = n_node.advertise<std_msgs::Bool>("mujoco_ros/mujoco_ros_interface/sim_run", 5);

    joint_states_pub_ = n_node.advertise<sensor_msgs::JointState>("joint_states", 5);    
    wrench_mesured_pub_ = n_node.advertise<geometry_msgs::Wrench>("wrench_measured", 5);
    
    /////////////// robot - ctrl(phri_hqp), robot(robot_wrapper) ///////////////        
    ee_state_pub_ = n_node.advertise<geometry_msgs::Transform>("mujoco_ros/mujoco_ros_interface/ee_state", 5);
    smach_pub_ = n_node.advertise<std_msgs::String>("kimm_polaris3d/state_transition", 5);
    
    // msg 
    robot_command_msg_.torque.resize(9);         // robot (7) + gripper(2) --> from here to mujoco    
    ee_state_msg_ = geometry_msgs::Transform();  // obtained from "robot_->position"

    sim_run_msg_.data = true;
    isgrasp_ = false;

    // ************ object estimation *************** //               
    parameter_init();
    kimm_polaris3d_smach_msg_.data = "pick_and_place";
    aruco_flag_ = true;
    yolo_flag_ = true;
    // ********************************************** //    

    while (ros::ok()){        
        //mujoco sim run 
        mujoco_run_pub_.publish(sim_run_msg_);
       
        //keyboard
        keyboard_event();

        // ctrl computation
        ctrl_->compute(time_); //make control input for 1kHz, joint state will be updated 1kHz from the mujoco
        
        // get output
        ctrl_->mass(robot_mass_);
        ctrl_->nle(robot_nle_);
        ctrl_->g(robot_g_);  // dim model.nv, [Nm]        
        ctrl_->state(state_);           
        ctrl_->JWorld(robot_J_world_);     //world
        
        // ctrl_->g_joint7(robot_g_local_);  //g [m/s^2] w.r.t joint7 axis
        // ctrl_->JLocal(robot_J_local_);     //local
        // ctrl_->dJLocal(robot_dJ_local_);   //local
        ctrl_->g_local_offset(robot_g_local_);    //local
        ctrl_->JLocal_offset(robot_J_local_);     //local
        ctrl_->dJLocal_offset(robot_dJ_local_);   //local

        // get control input from hqp controller
        ctrl_->franka_output(franka_qacc_); //get control input
        franka_torque_ = robot_mass_ * franka_qacc_ + robot_nle_;      

        // set control input to mujoco
        setGripperCommand();                              //set gripper torque by trigger value
        setRobotCommand();                                //set franka and husky command 
        robot_command_pub_.publish(robot_command_msg_);   //pub total command
       
        // get state
        getEEState();                                     //obtained from "robot_->position", and publish for monitoring              

        // ************ object estimation *************** //               
        FT_measured_pub();        
        // ********************************************** //    
        
        ros::spinOnce();
        loop_rate.sleep();        
    }//while

    return 0;
}

void FT_measured_pub() {
    //actually, franka_torque_ is not a measured but command torque, because measurment is not available
    tau_estimated = robot_mass_ * ddq_mujoco + robot_nle_;        
    // tau_ext = franka_torque_ - tau_estimated;      // coincide with g(0,0,-9.81)  
    tau_ext = -franka_torque_ + tau_estimated;        // coincide with g(0,0,9.81)
    FT_measured = robot_J_local_.transpose().completeOrthogonalDecomposition().pseudoInverse() * tau_ext;  //robot_J_local is local jacobian      

    geometry_msgs::Wrench FT_measured_msg;  
    FT_measured_msg.force.x = saturation(FT_measured[0],50);
    FT_measured_msg.force.y = saturation(FT_measured[1],50);
    FT_measured_msg.force.z = saturation(FT_measured[2],50);
    FT_measured_msg.torque.x = saturation(FT_measured[3],10);
    FT_measured_msg.torque.y = saturation(FT_measured[4],10);
    FT_measured_msg.torque.z = saturation(FT_measured[5],10);
    wrench_mesured_pub_.publish(FT_measured_msg);    
}

void parameter_init(){
    n_param = 10;
    m_FT = 6;
     
    FT_measured.resize(m_FT);    
    ddq_mujoco.resize(7);
    tau_estimated.resize(7);
    tau_ext.resize(7);
    v_mujoco.resize(6);
    a_mujoco.resize(6);
    a_mujoco_filtered.resize(6);      
    Fext_cali_.resize(7);
   
    FT_measured.setZero();   
    ddq_mujoco.setZero();
    tau_estimated.setZero();
    tau_ext.setZero();
    v_mujoco.setZero();
    a_mujoco.setZero();
    a_mujoco_filtered.setZero();   
    Fext_cali_.setZero();
}  

double saturation(double x, double limit) {
    if (x > limit) return limit;
    else if (x < -limit) return -limit;
    else return x;
}
// ************************************************ object estimation end *************************************************** //  

void ArucoCallback(const geometry_msgs::Pose &msg){
    if (!aruco_flag_){
        ctrl_->get_aruco_marker(msg);
        aruco_flag_ = true;
    } 
}

// void YoloCallback(const gb_visual_detection_3d_msgs::BoundingBoxes3d &msg){
//     if (!yolo_flag_){
//         if (msg.bounding_boxes[0].Class == "bottle") {
//             double x, y, z;
//             x = (msg.bounding_boxes[0].xmax + msg.bounding_boxes[0].xmin) /2.0;
//             y = (msg.bounding_boxes[0].ymax + msg.bounding_boxes[0].ymin) /2.0;
//             z = (msg.bounding_boxes[0].zmax + msg.bounding_boxes[0].zmin) /2.0;
            
//             cout << "x   " << x << endl;
//             cout << "y   " << y << endl;
//             cout << "z   " << z << endl;

//             geometry_msgs::Pose pos;
//             pos.position.x = x;
//             pos.position.y = y;
//             pos.position.z = z;
//             pos.orientation.x = 0.0;
//             pos.orientation.y = 0.0;
//             pos.orientation.z = 0.0;
//             pos.orientation.w = 1.0;

//             ctrl_->get_aruco_marker(pos);

//             yolo_flag_ = true;
//         }        
//     } 
// }

void YoloCallback(const visualization_msgs::MarkerArray &msg){
    if (!yolo_flag_){
            geometry_msgs::Pose pos;
            pos.position.x = msg.markers[0].pose.position.x;
            pos.position.y = msg.markers[0].pose.position.y;
            pos.position.z = msg.markers[0].pose.position.z;
            pos.orientation.x = ee_state_msg_.rotation.x;
            pos.orientation.y = ee_state_msg_.rotation.y;
            pos.orientation.z = ee_state_msg_.rotation.z;
            pos.orientation.w = ee_state_msg_.rotation.w;

            ctrl_->get_yolo_marker(pos);

            yolo_flag_ = true;               
    } 
}

void simCommandCallback(const std_msgs::StringConstPtr &msg){
    std::string buf;
    buf = msg->data;

    if (buf == "RESET")
    {
        std_msgs::String rst_msg_;
        rst_msg_.data = "RESET";
        mujoco_command_pub_.publish(rst_msg_);
    }

    if (buf == "INIT")
    {
        std_msgs::String rst_msg_;
        rst_msg_.data = "INIT";
        mujoco_command_pub_.publish(rst_msg_);
        mujoco_time_ = 0.0;
    }
}

void simTimeCallback(const std_msgs::Float32ConstPtr &msg){
    mujoco_time_ = msg->data;
    time_ = mujoco_time_;
}

void JointStateCallback(const sensor_msgs::JointState::ConstPtr& msg){ 
    // from mujoco
    // msg.position : 7(joint) + 2(gripper)
    // msg.velocity : 7(joint) + 2(gripper)
    sensor_msgs::JointState msg_tmp;
    msg_tmp = *msg;    

    //update state to pinocchio
    // state_.q_      //7 franka (7)
    // state_.v_      //7
    // state_.dv_     //7
    // state_.torque_ //7  

    ctrl_->franka_update(msg_tmp);        
    joint_states_publish(msg_tmp);        

    v_mujoco.setZero();
    a_mujoco.setZero();    
    for (int i=0; i<7; i++){ 
        ddq_mujoco[i] = msg_tmp.effort[i];
        v_mujoco += robot_J_local_.col(i) * msg_tmp.velocity[i];                                                 //jacobian is LOCAL
        a_mujoco += robot_dJ_local_.col(i) * msg_tmp.velocity[i] + robot_J_local_.col(i) * msg_tmp.effort[i];    //jacobian is LOCAL        
    }        

    // Filtering
    double cutoff = 20.0; // Hz //20
    double RC = 1.0 / (cutoff * 2.0 * M_PI);    
    double alpha = dt / (RC + dt);

    a_mujoco_filtered = alpha * a_mujoco + (1 - alpha) * a_mujoco_filtered;            
}

void joint_states_publish(const sensor_msgs::JointState& msg){
    // mujoco callback msg
    // msg.position : 7(joint) + 6(gripper)
    // msg.velocity : 7(joint) + 6(gripper) 

    sensor_msgs::JointState joint_states;
    joint_states.header.stamp = ros::Time::now();    

    //revolute joint name in rviz urdf (panda_arm_2f_85_d435_rviz.urdf)
    joint_states.name = {"panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7","finger_joint","left_inner_knuckle_joint", "left_inner_finger_joint", "right_inner_knuckle_joint", "right_inner_finger_joint", "right_outer_knuckle_joint"};    

    joint_states.position.resize(13); //panda(7) + finger(6)
    joint_states.velocity.resize(13); //panda(7) + finger(6)

    for (int i=0; i<13; i++){ 
        joint_states.position[i] = msg.position[i];
        joint_states.velocity[i] = msg.velocity[i];
    }    

    joint_states_pub_.publish(joint_states);    
}

void ctrltypeCallback(const std_msgs::Int16ConstPtr &msg){
    ROS_WARN("%d", msg->data);
    
    if (msg->data != 899){
        int data = msg->data;
        ctrl_->ctrl_update(data);
    }
    else{
        if (isgrasp_)
            isgrasp_=false;
        else
            isgrasp_=true;
    }
}

void setRobotCommand(){
    robot_command_msg_.MODE = 1; //0:position control, 1:torque control
    robot_command_msg_.header.stamp = ros::Time::now();
    robot_command_msg_.time = time_;   
    
    for (int i=0; i<7; i++)
        robot_command_msg_.torque[i] = franka_torque_(i);    
}

void setGripperCommand(){
    if (isgrasp_){
        robot_command_msg_.torque[7] = -200.0;
        robot_command_msg_.torque[8] = -200.0;
    }
    else{
        robot_command_msg_.torque[7] = 100.0;
        robot_command_msg_.torque[8] = 100.0;
    }
}

void getEEState(){
    Vector3d pos;
    Quaterniond q;
    ctrl_->ee_state(pos, q);

    ee_state_msg_.translation.x = pos(0);
    ee_state_msg_.translation.y = pos(1);
    ee_state_msg_.translation.z = pos(2);

    ee_state_msg_.rotation.x = q.x();
    ee_state_msg_.rotation.y = q.y();
    ee_state_msg_.rotation.z = q.z();
    ee_state_msg_.rotation.w = q.w();
    ee_state_pub_.publish(ee_state_msg_);
}

void keyboard_event(){
    if (_kbhit()){
        int key;
        key = getchar();
        int msg = 0;
        switch (key){
            case 'g': //gravity mode
                msg = 0;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "Gravity mode" << endl;
                cout << " " << endl;
                break;
            case 'h': //home
                msg = 1;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "home position" << endl;
                cout << " " << endl;
                break;

            case 'x': //aruco marker save
                aruco_flag_ = false;                      
                cout << " " << endl;
                cout << "aruco marker save" << endl;
                cout << " " << endl;
                break;               
            
            case 'q': //init position w.r.t ereon
                msg = 11;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "init position w.r.t ereon" << endl;
                cout << " " << endl;
                break;  
            case 'w': //transition pos to place bottle
                msg = 12;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "transition pos to place bottle" << endl;
                cout << " " << endl;
                break; 
            case 'e': //go to ereon plate to place bottle
                msg = 13;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "go to ereon plate to place bottle" << endl;
                cout << " " << endl;
                break;                                                   
            
            case 'a': //init position w.r.t. bottle
                msg = 21;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "init position w.r.t. bottle" << endl;
                cout << " " << endl;
                break; 
            case 's': //transition pos to pick bottle
                msg = 22;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "transition pos to pick bottle" << endl;
                cout << " " << endl;
                break; 
            case 'd': //bottle pick pos
                msg = 23;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "bottle pick pos" << endl;
                cout << " " << endl;
                break;                
                            
            // case 'v': //impedance control
            //     msg = 22;
            //     ctrl_->ctrl_update(msg);
            //     cout << " " << endl;
            //     cout << "impedance control" << endl;
            //     cout << " " << endl;
            //     break;   

            case 'i': //move ee +0.1z
                msg = 31;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "move ee +0.1 z" << endl;
                cout << " " << endl;
                break;         
            case 'k': //move ee -0.1z
                msg = 32;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "move ee -0.1 z" << endl;
                cout << " " << endl;
                break;                                   
                break;         
            case 'l': //move ee +0.1x
                msg = 33;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "move ee +0.1 x" << endl;
                cout << " " << endl;
                break;                                   
            case 'j': //move ee -0.1x
                msg = 34;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "move ee -0.1 x" << endl;
                cout << " " << endl;
                break;      
            case 'u': //move ee -0.1y
                msg = 35;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "move ee -0.1 y" << endl;
                cout << " " << endl;
                break;              
            case 'o': //move ee +0.1y
                msg = 36;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "move ee +0.1 y" << endl;
                cout << " " << endl;
                break;  

            case ']': //start pick and place process with smach
                smach_pub_.publish(kimm_polaris3d_smach_msg_);                
                cout << " " << endl;
                cout << "start pick and place process with smach" << endl;
                cout << " " << endl;
                break;  

            case 'p': //print current EE state
                msg = 99;
                ctrl_->ctrl_update(msg);
                cout << " " << endl;
                cout << "print current EE state" << endl;
                cout << " " << endl;
                break;     

            case 'z': //grasp
                if (isgrasp_){
                    cout << "Release hand" << endl;
                    isgrasp_ = false;
                }
                else{
                    cout << "Grasp object" << endl;
                    isgrasp_ = true; 
                }
                break;
        }
    }
}

