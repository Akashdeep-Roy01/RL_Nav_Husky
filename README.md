# RL_Nav_Husky
![ROS2](https://camo.githubusercontent.com/b874e7cbc7323284002070083cf5fc1cfff41a3a5573f598638564fa4017fd65/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f524f5320322d68756d626c652d626c756576696f6c6574)
![Docker](https://camo.githubusercontent.com/b609225bdb4ad668a23ec18b022f166bd86e184b28ddcf34b604f4a68837907f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f636b65722d3234393645443f7374796c653d666c61742d737175617265266c6f676f3d646f636b6572266c6f676f436f6c6f723d7768697465)

Repository to develop and test different deep Reinforcement Learning based algorithms for Autonomous Robot Navigation.

Description

The `rl_nav_description` package contains -
1. Urdf files related to Clearpath Husky robot and Ouster LiDAR Sensor.
2. Launch file to launch the Gazebo and Rviz nodes.

The `rl_nav_bringup` contains -
1. Files that define the environment as per OpenAI gym guidelines and is compatible with StableBaselines3
2. Files related to parameter tuning, training and testing.

To use:

1. Build the docker file using `docker build -t <image name> .`
2. Change the image name inside "docker_run.sh" and run it using `bash docker_run.sh`
3. Build the package using `colcon build --symlink-install --packages-select rl_nav_description`
4. Source using `source /docker_ws/install/setup.bash`
5. Launch the robot simulation using `ros2 launch rl_nav_description launch_simulation.launch.py`
6. To train a new model first determine the optimal hyperparameters by running `python3 src/rl_nav_bringup/SAC_hyperparameter_tuning.py`
7. Run `python3 src/rl_nav_bringup/train_SAC.py` after modifying the hyperparameters inside.
8. To evaluate run `python3 src/rl_nav_bringup/test.py` after modifying the path to the saved model.

Results

As can be seen the robot tries to reach the randomly spawned goal (indicated by the green dot) while avoiding obstacles detected using LiDAR. It was trained using SAC algorithm. 


https://github.com/Akashdeep-Roy01/RL_Nav_Husky/assets/99131809/d850547c-2da4-47e4-a73a-4ea63f766bae

