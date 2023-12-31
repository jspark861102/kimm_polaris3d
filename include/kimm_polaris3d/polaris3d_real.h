#pragma once

// controller
#include "kimm_polaris3d/polaris3d_hqp.h"

// gazebo-like controller interface
#include <controller_interface/multi_interface_controller.h>
#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <dynamic_reconfigure/server.h>

// Ros
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float32.h"
#include "sensor_msgs/JointState.h"
#include "geometry_msgs/Transform.h"
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include <ros/package.h>
#include <realtime_tools/realtime_publisher.h>
#include "mujoco_ros_msgs/JointSet.h"
#include "visualization_msgs/Marker.h"

// tf
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

// Franka
#include <franka/robot_state.h>
#include <franka_gripper/GraspAction.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/trigger_rate.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <franka_gripper/GraspAction.h>
#include <franka_gripper/MoveAction.h>
#include <franka_msgs/SetLoad.h>

//robotiq gripper
#include <robotiq_2f_gripper_msgs/CommandRobotiqGripperAction.h>
#include <robotiq_2f_gripper_control/robotiq_gripper_client.h>

// System
#include <iostream>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 7, 7> Matrix7d;

namespace kimm_franka_controllers {

class BasicFrankaController : public controller_interface::MultiInterfaceController<
								   franka_hw::FrankaModelInterface,
								   hardware_interface::EffortJointInterface,
								   franka_hw::FrankaStateInterface> {
                     
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;
  void stopping(const ros::Time& /*time*/) override;

  void ctrltypeCallback(const std_msgs::Int16ConstPtr &msg);  
  void ArucoCallback(const geometry_msgs::Pose &msg);


  void asyncCalculationProc(); 
  void modeChangeReaderProc();
  void setFrankaCommand();
  void pubEEState();  
  void pubJointStates();

  void robotiqstateCallback(const sensor_msgs::JointStateConstPtr &msg);

  void keyboard_event();
  bool _kbhit()
  {
      termios term;
      tcgetattr(0, &term);

      termios term2 = term;
      term2.c_lflag &= ~ICANON;
      tcsetattr(0, TCSANOW, &term2);

      int byteswaiting;
      ioctl(0, FIONREAD, &byteswaiting);

      tcsetattr(0, TCSANOW, &term);

      return byteswaiting > 0;
  }
  double lowpassFilter(double sample_time, double y, double y_last, double cutoff_frequency){ //cutoff_frequency is in [Hz]
      double gain = sample_time / (sample_time + (1.0 / (2.0 * M_PI * cutoff_frequency)));
      return gain * y + (1 - gain) * y_last;
  }
  Vector7d lowpassFilter(double sample_time, const Vector7d y, const Vector7d y_last, double cutoff_frequency){ //cutoff_frequency is in [Hz]
      double gain = sample_time / (sample_time + (1.0 / (2.0 * M_PI * cutoff_frequency)));
      return gain * y + (1 - gain) * y_last;
  }
  Vector6d lowpassFilter(double sample_time, const Vector6d y, const Vector6d y_last, double cutoff_frequency){ //cutoff_frequency is in [Hz]
      double gain = sample_time / (sample_time + (1.0 / (2.0 * M_PI * cutoff_frequency)));
      return gain * y + (1 - gain) * y_last;
  }  

 private: 
    ros::Publisher ee_state_pub_, torque_state_pub_, joint_state_pub_, smach_pub_;

    geometry_msgs::Transform ee_state_msg_;
    std_msgs::Float32 time_msg_;
    mujoco_ros_msgs::JointSet robot_command_msg_;
    sensor_msgs::JointState robot_state_msg_;
    franka_msgs::SetLoad setload_srv;
    sensor_msgs::JointState robotiq_state_msg_;

    // thread
    std::mutex calculation_mutex_;
    std::thread async_calculation_thread_, mode_change_thread_;
    bool is_calculated_{false}, quit_all_proc_{false};

    std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
    std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
    std::vector<hardware_interface::JointHandle> joint_handles_;
    std::string group_name_;    

    robotiq_2f_gripper_control::RobotiqActionClient * gripper_robotiq_;        

    //franka_gripper::GraspGoal goal;
    franka_hw::TriggerRate print_rate_trigger_{100}; 
    Eigen::Matrix<double, 7, 1> saturateTorqueRate(const Eigen::Matrix<double, 7, 1>& tau_d_calculated, const Eigen::Matrix<double, 7, 1>& tau_J_d);

    // Variables
    const double delta_tau_max_{30.0};
    Vector7d dq_filtered_, franka_torque_, franka_q_, franka_dq_, franka_dq_d_;
    VectorXd franka_ddq_;
    Vector6d f_filtered_, f_;
    Eigen::VectorXd franka_qacc_, robot_nle_, robot_g_;
    MatrixXd robot_mass_, robot_J_, robot_tau_, robot_tau_d_;
    double time_, dt_;
    bool isgrasp_, aruco_flag_;
    std_msgs::String kimm_polaris3d_smach_msg_;
    
    State state_;    

    RobotController::FrankaWrapper * ctrl_;
    ros::Subscriber ctrl_type_sub_, robotiq_state_subs_, aruco_sub_;
};

}  // namespace kimm_franka_controllers
