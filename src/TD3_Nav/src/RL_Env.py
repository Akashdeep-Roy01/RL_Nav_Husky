import os
import copy
import argparse
import numpy as np
import rclpy
import time
import torch
import math
import threading
from TD3 import *
from rclpy.node import Node
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from common.utils import *
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import xml.etree.ElementTree as ET

class RLEnv(Node):
    def __init__(self):
        super().__init__('env')

        # Define constants
        self.collision_distance = 0.80 # in meters
        self.goal_reach_distance = 0.50 # in meters
        self.margin = 1.0 # radius of margin[in meters] around obstacles
        self.area_limits = 6.5 # defines length of env in x & y
        
        # Inialise different variables
        self.robot_x = 0.0 # Robot positions 
        self.robot_y = 0.0        
        self.goal_x = 0.0 # goal locations
        self.goal_y = 0.0    
        self.upper = 3.5 # to enable easy goals in the beginning
        self.lower = -3.5     
        self.initial_distance=0.0

        # Define topic names
        self.velocity_topic = "/cmd_vel"
        self.goal_topic = "/goal_pose"

        self.obstacle_list = ["Construction Barrel","Construction Barrel_0","Construction Barrel_1"] # For maze5.sdf
        self.vel_pub = self.create_publisher(Twist,self.velocity_topic,10)
        self.publisher = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_simulation_client = self.create_client(Empty, 'reset_world')
        self.set_state = self.create_client(SetEntityState, 'set_entity_state')
                
    def check_pos(self,x,y):

        tree = ET.parse("/docker_ws/src/rl_nav_description/worlds/maze6.sdf")
        obstacle_coordinates=[]
        root=tree.getroot()
        all = root[0].findall("model")
        for a in range(1,len(all)):
            temp_name = all[a].attrib["name"]
            if temp_name not in self.obstacle_list:
                pose = all[a].find("pose").text.split()
                size = all[a].find("link").find("collision").find("geometry").find("box").find("size").text.split()
                pose_x = float(pose[0])
                pose_y = float(pose[1])
                rotation = float(pose[-1])
                if rotation == 0:
                    size_x = float(size[0])+self.margin*2
                    size_y = float(size[1])+self.margin*2
                else:
                    size_x = float(size[1])+self.margin*2
                    size_y = float(size[0])+self.margin*2

                point_1 = [pose_x + size_x / 2, pose_y + size_y / 2]
                point_2 = [point_1[0], point_1[1] - size_y]
                point_3 = [point_1[0] - size_x, point_1[1] - size_y ]
                point_4 = [point_1[0] - size_x, point_1[1] ]
                area_points = [point_1, point_2, point_3, point_4]
                obstacle_coordinates.append(area_points)
                # print(area_points)

                xmin=min(point_1[0],point_2[0],point_3[0],point_4[0])
                xmax=max(point_1[0],point_2[0],point_3[0],point_4[0])
                ymin=min(point_1[1],point_2[1],point_3[1],point_4[1])
                ymax=max(point_1[1],point_2[1],point_3[1],point_4[1])

                if xmax>x>xmin and ymax>y>ymin:
                    return False
                if abs(x)>8.0 or abs(y)>8.0:
                    return False
                
        return True
     
    def shuffle_goal(self):
        if self.upper < self.area_limits:
            self.upper += 0.004
        if self.lower > -self.area_limits:
            self.lower -= 0.004
        goal_ok = False
        while not goal_ok:
            x_add = np.random.uniform(self.upper,self.lower)
            y_add = np.random.uniform(self.upper,self.lower)    
            self.goal_x = self.robot_x + x_add
            self.goal_y = self.robot_y + y_add
            goal_ok = self.check_pos(self.goal_x,self.goal_y)
   
    def shuffle_robot(self):
        name = "my_bot"
        angle = np.random.uniform(-np.pi,np.pi)
        self.robot_yaw = get_quaternion_from_euler(0,0,angle)
        pos_ok = False
        while not pos_ok:
            self.robot_x = np.random.uniform(-self.area_limits, self.area_limits)
            self.robot_y = np.random.uniform(-self.area_limits, self.area_limits)
            pos_ok = self.check_pos(self.robot_x, self.robot_y)
        req = SetEntityState.Request()
        req._state.name = name
        req._state.pose.position.x = self.robot_x
        req._state.pose.position.y = self.robot_y
        req._state.pose.orientation.x = self.robot_yaw[0]
        req._state.pose.orientation.y = self.robot_yaw[1]
        req._state.pose.orientation.z = self.robot_yaw[2]
        req._state.pose.orientation.w = self.robot_yaw[3]

        self.move_entity(req)

    def shuffle_obstacles(self):
        
        for name in self.obstacle_list:
            pos_ok = False
            while not pos_ok:
                obstacle_x = np.random.uniform(-self.area_limits, self.area_limits)
                obstacle_y = np.random.uniform(-self.area_limits, self.area_limits)
                pos_ok = self.check_pos(obstacle_x, obstacle_y)
                distance_to_robot = np.linalg.norm([obstacle_x - self.robot_x, obstacle_y - self.robot_y])
                distance_to_goal = np.linalg.norm([obstacle_x - self.goal_x, obstacle_y - self.goal_y])
                if distance_to_robot < 3.0 or distance_to_goal < 3.0:
                    pos_ok = False
            req = SetEntityState.Request()
            req._state.name = name
            req._state.pose.position.x = obstacle_x
            req._state.pose.position.y = obstacle_y
            self.move_entity(req)

    def move_entity(self,req):
        try:
            self.set_state.call_async(req)
        except rclpy.ServiceException as e:
            print("set_entity_state service call failed")   

    def reset(self):
        req = Empty.Request()

        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            print('reset service not available, waiting again...')
        try:
            self.reset_simulation_client.call_async(req)
        except:
            print("/gazebo/reset_simulation service call failed")

        self.shuffle_robot()
        self.shuffle_goal()
        self.publish_markers()
        # self.shuffle_obstacles()

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(1)
        while not self.pause.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")

        laser_state = copy.deepcopy(ouster_data)
        diff_x = self.robot_x - self.goal_x
        diff_y = self.robot_y - self.goal_y
        self.initial_distance = math.sqrt(diff_x**2 + diff_y**2)
        robot_state = [self.initial_distance,0.0,0.0]
        state = np.append(laser_state,robot_state)

        return state

    def step(self, action):
        success = False
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        self.publish_markers()        
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(0.1)
        while not self.pause.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")
        laser_state = copy.deepcopy(ouster_data)
        done, collision, min_laser = self.check_collision(laser_state)
        self.robot_x = last_odom.pose.pose.position.x
        self.robot_y = last_odom.pose.pose.position.y
        self.robot_yaw[0] = last_odom.pose.pose.orientation.x
        self.robot_yaw[1] = last_odom.pose.pose.orientation.y
        self.robot_yaw[2] = last_odom.pose.pose.orientation.z
        self.robot_yaw[3] = last_odom.pose.pose.orientation.w
        
        diff_x = self.robot_x - self.goal_x
        diff_y = self.robot_y - self.goal_y
        distance = math.sqrt(diff_x**2 + diff_y**2)
        # Detect if the goal has been reached and give a large positive reward
        if distance < self.goal_reach_distance:
            print("Reached GOAL!")
            success = True
            done = True

        robot_state = [distance,action[0],action[1]]
        state = np.append(laser_state,robot_state)

        reward = self.get_reward(success, collision, action, distance, self.initial_distance, min_laser)
        return state,reward, done,collision

    def get_random_action(self):
        random_linear = np.random.uniform(0, 1)
        random_angular = np.random.uniform(-1, 1)
        random_action = [random_linear, random_angular]
        return random_action

    def publish_markers(self):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = float(self.goal_reach_distance/2)
        marker.scale.y = float(self.goal_reach_distance/2)
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0.0
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)
   
    def check_collision(self,laser_data):
        min_dist = min(laser_data)
        if min_dist < self.collision_distance:
            return True,True,min_dist
        return False,False,min_dist
    
    @staticmethod
    def get_reward(succeed, collision, action, goal_dist, goal_dist_initial,min_obstacle_dist):
        if succeed:
            reward = 1000
        elif collision:
            reward = -1000
        else:
            r_vangular = -abs(action[1]) #[-1,0]
            r_distance = 10*((0.1 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 0.1) #[-1,0]
            reward = r_distance+r_vangular
        return float(reward)

class Odom_subscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.odom_topic = "/odom"
        self.subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10)

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

class Velodyne_subscriber(Node):
    def __init__(self):
        super().__init__('velodyne_subscriber')
        self.laser_topic = "/laser_controller/out"
        self.num_scan = 40 # number of bins for laser scan data
        self.scan_sub = self.create_subscription(PointCloud2, self.laser_topic,self.laser_callback,10)
        self.gaps = [[-np.pi, -np.pi + (2*np.pi)/self.num_scan]]
        for m in range(self.num_scan - 1):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + (2*np.pi)/ self.num_scan])

    def laser_callback(self, v):
        global ouster_data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        ouster_data = np.ones(self.num_scan) * 10
        for i in range(len(data)):
            if data[i][2] > 0:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        ouster_data[j] = min(ouster_data[j], dist)
                        break


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy,env,max_step, eval_episodes=10):
    eval_env = env
    avg_reward = 0.
    for i in range(eval_episodes):
        state, done = eval_env.reset(), False
        count = 0
        while not done and count<max_step:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            count+=1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    rclpy.init()    

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  #    Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)    # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)           # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)          # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)         # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)             # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)                  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                    # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates
    parser.add_argument("--buffer_size", default=2e5, type=int)         # Buffer Capacity
    parser.add_argument("--max_episode_steps", default=500, type=int)   # Episode Length		
    parser.add_argument("--save_model", action="store_true")            # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                     # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = RLEnv()
    laser = Velodyne_subscriber()
    odom = Odom_subscriber()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom)
    executor.add_node(laser)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    file_name = f"{args.policy}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    
    # Set seeds

    state_dim = 43
    action_dim = 2 
    max_action = 1.0

    policy = TD3(state_dim,action_dim,max_action,args.discount,args.tau,args.policy_noise * max_action,args.noise_clip * max_action,args.policy_freq)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayMemory(args.buffer_size, args.seed)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env, args.max_episode_steps)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.get_random_action()
            # print(action)
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
            # print(action)

        # Perform 
        # print(action)
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < args.max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.push(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if episode_timesteps >= args.max_episode_steps:
            done = True

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state= env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            done = False

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, env, args.max_episode_steps))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
    rclpy.shutdown()


