# kimm_phri_panda
KIMM pHRI application with Padna Robot Arm

## 1. Prerequisites
### 1.1 Robot controller
```bash
git clone https://github.com/jspark861102/kimm_qpoases.git -b melodic
git clone https://github.com/jspark861102/kimm_hqp_controller_phri.git -b melodic
git clone https://github.com/jspark861102/kimm_trajectory_smoother.git -b melodic
```

### 1.2 Robot model and simulator
```bash
#git clone https://github.com/jspark861102/franka_ros.git #my used version (0.8.1)
git clone https://github.com/frankaemika/franka_ros.git 
git clone https://github.com/jspark861102/robotiq_2finger_grippers.git
git clone https://github.com/jspark861102/kimm_robots_description.git -b melodic
git clone https://github.com/jspark861102/kimm_mujoco_ros.git -b melodic
```

### 1.3 visual perception (aruco marker & YOLO)
```bash
git clone https://github.com/jspark861102/kimm_aruco.git -b noetic-devel
git clone https://github.com/jspark861102/kimm_darknet_ros.git --recursive
```

### 1.4 polaris3d beverage delivery task
```bash
git clone https://github.com/jspark861102/kimm_polaris3d.git
```

## 2. Run
### 2.1 Simulation
```bash
# Simulation with PC Monitor
roslaunch kimm_polaris3d ns0_simulation.launch

# Simulation with 17inch Notebook
roslaunch kimm_polaris3d ns0_simulation.launch note_book:=true
```

### 2.1 Real Robot
```bash
# beverage delivery task
roslaunch kimm_polaris3d ns0_real_robot.launch
```
