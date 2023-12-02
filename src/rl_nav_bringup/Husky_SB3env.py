import copy
import numpy as np
import gymnasium as gym
from Husky_Robot import *
from common.utils import *
from gymnasium import spaces
from geometry_msgs.msg import Twist


class Husky_SB3Env(Husky_Robot,gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    def __init__(self):
        super().__init__()
        self.collision_distance = 0.80
        self.episode_steps = 0
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Dict(
                {
                "agent": spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float64),
                "laser": spaces.Box(low=0, high=1, shape=(self.num_scan,), dtype=np.float64),
                }
            )
        
    def reset(self,seed=None,options=None):

        self.episode_steps = 0
        self.reset_done = False
        self.reset_simulation()
        while self.reset == False:
            rclpy.spin_once(node=self)

        self.shuffle_robot()
        self.shuffle_goal()
        self.publish_markers()
        self.shuffle_obstacles()
        rclpy.spin_once(node=self,timeout_sec=1.0)
    
        laser_state = copy.deepcopy(self.ouster_data)
        laser_state = laser_state/10.0 # Normalise to make computations faster
        distance = self.get_target_distance()
        heading_angle = self.get_target_heading()
        robot_state = np.array([distance,heading_angle]) 
        self.prev_goal_dist = distance

        state = {
            "agent":robot_state,
            "laser":laser_state
        }

        info = {}
        
        return state, info

    def step(self, action):
        success = False

        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        self.publish_markers()
           
        rclpy.spin_once(node=self,timeout_sec=0.2)        

        laser_state = copy.deepcopy(self.ouster_data)
        done, collision, _ = self.check_collision(laser_state)
        laser_state = laser_state/10.0 # Normalise to make computations faster

        self.robot_x = self.last_odom.pose.pose.position.x
        self.robot_y = self.last_odom.pose.pose.position.y
        self.robot_yaw[0] = self.last_odom.pose.pose.orientation.x
        self.robot_yaw[1] = self.last_odom.pose.pose.orientation.y
        self.robot_yaw[2] = self.last_odom.pose.pose.orientation.z
        self.robot_yaw[3] = self.last_odom.pose.pose.orientation.w
        
        distance = self.get_target_distance()
        heading_angle = self.get_target_heading()

        # Detect if the goal has been reached and give a large positive reward
        if distance < self.goal_reach_distance:
            # print("Reached GOAL!")
            success = True
            done = True

        robot_state = np.array([distance,heading_angle])
        
        state = {
            "agent":robot_state,
            "laser":laser_state
        }

        reward = self.get_reward(success, collision, distance)

        self.prev_goal_dist = distance

        self.episode_steps+=1

        if self.episode_steps>=500:
            truncated = True
        else:
            truncated = False

        info = {}

        return state,reward,done,truncated,info
   
    def check_collision(self,laser_data):
        min_dist = min(laser_data)
        if min_dist < self.collision_distance:
            return True,True,min_dist
        return False,False,min_dist
    
    def get_reward(self, succeed, collision, goal_dist):
        if succeed:
            reward = 2000.0 - self.episode_steps # Range 1501 to 2000
        elif collision:
            reward = -2000.0
        else: # Range -1000.0 to 500.0
            if goal_dist < self.prev_goal_dist:
                reward = 1.0
            elif goal_dist > self.prev_goal_dist:
                reward = -2.0
            else:
                reward = 0.0
        return float(reward)
    
    def render(self):
        pass

    def close(self):
        self.destroy_node()

