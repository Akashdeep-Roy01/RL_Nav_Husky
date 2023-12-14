import math
import rclpy
import numpy as np
from common.utils import *
from rclpy.node import Node
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from gazebo_msgs.srv import SetEntityState
from visualization_msgs.msg import MarkerArray
from sensor_msgs_py import point_cloud2 as pc2

class Husky_Robot(Node):
    def __init__(self):
        super().__init__("husky_bot")   
        # self.get_logger().info("The robot controller node has just been created")  
        self.robot_x = 0.0 # Robot positions 
        self.robot_y = 0.0        
        self.goal_x = 0.0 # goal locations
        self.goal_y = 0.0    
        self.upper = 1.5 # to enable easy goals in the beginning
        self.lower = -1.5     
        self.goal_reach_distance = 0.0278 # normalised (divide by max.possible 18) in meters        
        self.num_scan = 40 # number of bins for laser scan data
        self.area_limits = 8.0 # defines length of env in x & y
        
        self.set_state = self.create_client(SetEntityState, 'set_entity_state')        
        self.publisher = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)
        
        self.obstacle_list = []
        margin = 1.0 # radius of margin[in meters] around obstacles
        world_file = "/docker_ws/src/rl_nav_description/worlds/maze7.sdf"
        self.occupied_area = get_occupied_area(world_file=world_file,\
                                               obstacle_list=self.obstacle_list,margin=margin)

        self.velocity_topic = "/cmd_vel"        
        self.vel_pub = self.create_publisher(Twist,self.velocity_topic,10)

        self.odom_topic = "/odom"
        self.subscription = self.create_subscription(Odometry,self.odom_topic,self.odom_callback,1)

        self.laser_topic = "/laser_controller/out"
        self.scan_sub = self.create_subscription(PointCloud2, self.laser_topic,self.laser_callback,1)
        self.ouster_data = np.ones(self.num_scan) * 10.0
        self.gaps = [[-np.pi, -np.pi + (2*np.pi)/self.num_scan]]
        for m in range(self.num_scan - 1):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + (2*np.pi)/ self.num_scan])
        
        self.reset_simulation_client = self.create_client(Empty, 'reset_world')


    def laser_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.ouster_data = np.ones(self.num_scan) * 10.0
        for i in range(len(data)):
            if data[i][2] > 0:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.ouster_data[j] = min(self.ouster_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def check_pos(self,x,y):
        """
            Function to check if spawn position is valid i.e. not near or on top of any obstacle
        """
        if abs(x)>=8.0 or abs(y)>=8.0:
            return False
        for ob in self.occupied_area:
            xmin=ob[0]
            xmax=ob[1]
            ymin=ob[2]
            ymax=ob[3]
            if xmax>=x>=xmin and ymax>=y>=ymin:
                return False                                
        return True
     
    def shuffle_goal(self):
        # if self.upper < self.area_limits:
        #     self.upper += 0.004
        # if self.lower > -self.area_limits:
        #     self.lower -= 0.004
        goal_ok = False
        while not goal_ok:
            # x_add = np.random.uniform(self.upper,self.lower)
            # y_add = np.random.uniform(self.upper,self.lower)    
            x_add = np.random.uniform(-self.area_limits,self.area_limits)
            y_add = np.random.uniform(-self.area_limits,self.area_limits)
            self.goal_x = self.robot_x + x_add
            self.goal_y = self.robot_y + y_add
            goal_ok = self.check_pos(self.goal_x,self.goal_y)
        # print("Goal Changed",self.goal_x,self.goal_y)
   
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
    
    def reset_simulation(self):
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            print('reset service not available, waiting again...')
        try:
            self.reset_simulation_client.call_async(Empty.Request())
            self.reset_done=True
        except:
            print("/gazebo/reset_simulation service call failed")

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
        except rclpy.ServiceException:
            print("set_entity_state service call failed") 

    def publish_markers(self):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
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
        
    def get_target_distance(self):
        # print(self.robot_x,self.robot_y,self.goal_x,self.goal_y)
        distance = np.linalg.norm([self.robot_x-self.goal_x, self.robot_y-self.goal_y])
        distance = distance / 18.0 # Normalise to make computaions faster inside the network
        return distance
    
    def get_target_heading(self):
        theta=euler_from_quaternion(self.robot_yaw)[-1]
        robot_target_diff_y = math.sin(-theta)*(self.goal_x-self.robot_x)+math.cos(theta)*(self.goal_y-self.robot_y)
        robot_target_diff_x = math.cos(theta)*(self.goal_x-self.robot_x)+math.sin(theta)*(self.goal_y-self.robot_y)
        heading = math.atan2(robot_target_diff_y,robot_target_diff_x)
        heading = heading/(2*math.pi) # Normalise to make computaions faster inside the network
        return heading