// Controller
#include "kimm_polaris3d/polaris3d_hqp.h"

//Mujoco MSG Header
#include "mujoco_ros_msgs/JointSet.h"
#include "mujoco_ros_msgs/SensorState.h"

// Ros
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Bool.h"
#include "sensor_msgs/JointState.h"
#include "geometry_msgs/Transform.h"
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Wrench.h"
#include "geometry_msgs/Pose.h"

// Tf
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

// #include <gb_visual_detection_3d_msgs/BoundingBoxes3d.h>
#include <visualization_msgs/MarkerArray.h>

ros::Publisher mujoco_command_pub_;
ros::Publisher robot_command_pub_;
ros::Publisher mujoco_run_pub_;
ros::Publisher joint_states_pub_, wrench_mesured_pub_;
ros::Publisher ee_state_pub_;
ros::Publisher smach_pub_;

mujoco_ros_msgs::JointSet robot_command_msg_;
geometry_msgs::Transform ee_state_msg_;
sensor_msgs::JointState base_state_msg_;

double mujoco_time_, time_, dt;
bool isgrasp_;
Eigen::VectorXd franka_qacc_, husky_qacc_, robot_nle_, robot_g_, franka_torque_, Fext_cali_;
Eigen::MatrixXd robot_mass_, robot_J_local_, robot_dJ_local_, robot_J_world_;
string group_name;
std_msgs::String sim_run_msg_;
std_msgs::String kimm_polaris3d_smach_msg_;

RobotController::FrankaWrapper * ctrl_;
State state_;

// void YoloCallback(const gb_visual_detection_3d_msgs::BoundingBoxes3d &msg);
void YoloCallback(const visualization_msgs::MarkerArray &msg);
void ArucoCallback(const geometry_msgs::Pose &msg);
void simCommandCallback(const std_msgs::StringConstPtr &msg);
void simTimeCallback(const std_msgs::Float32ConstPtr &msg);
void JointStateCallback(const sensor_msgs::JointState::ConstPtr& msg);
void ctrltypeCallback(const std_msgs::Int16ConstPtr &msg);
void joint_states_publish(const sensor_msgs::JointState& msg);

void setRobotCommand();
void setGripperCommand();
void getEEState();

bool isFextapplication_, isFextcalibration_, aruco_flag_, yolo_flag_;
double n_param, m_FT;
Eigen::VectorXd FT_measured, robot_g_local_;
VectorXd ddq_mujoco, tau_estimated, tau_ext, v_mujoco, a_mujoco, a_mujoco_filtered;

void parameter_init();
double saturation(double x, double limit);
void FT_measured_pub();

void EEStatusCallback(geometry_msgs::Transform& msg, const pinocchio::SE3 & H_ee);

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
};

