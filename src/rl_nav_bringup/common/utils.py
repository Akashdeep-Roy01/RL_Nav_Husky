import numpy as np
import xml.etree.ElementTree as ET

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qx, qy, qz, qw]

def euler_from_quaternion(quat):
  """
  Converts quaternion (w in last place) to euler roll, pitch, yaw
  quat = [x, y, z, w]
  """
  x = quat[0]
  y = quat[1]
  z = quat[2]
  w = quat[3]

  sinr_cosp = 2 * (w*x + y*z)
  cosr_cosp = 1 - 2*(x*x + y*y)
  roll = np.arctan2(sinr_cosp, cosr_cosp)

  sinp = 2 * (w*y - z*x)
  if sinp < -1:
    sinp = -1
  if sinp > 1:
    sinp = 1
  pitch = np.arcsin(sinp)

  siny_cosp = 2 * (w*z + x*y)
  cosy_cosp = 1 - 2 * (y*y + z*z)
  yaw = np.arctan2(siny_cosp, cosy_cosp)

  return roll, pitch, yaw

def get_occupied_area(world_file,obstacle_list,margin):
  """
  Takes a world file and returns the coordinates of regions occupied by static obstacles
  Input
    :world_file:path to thw world_file as a string
    :obstcale_list:list of obstacles to avoid i.e. obstacles that may not be static and may change location
    :margin:safety margin around obstacles as a float
  """
  tree = ET.parse(world_file)
  obstacle_coordinates=[]
  root=tree.getroot()
  all = root[0].findall("model")
  for a in range(1,len(all)):
    temp_name = all[a].attrib["name"]
    if temp_name not in obstacle_list:
      pose = all[a].find("pose").text.split()
      size = all[a].find("link").find("collision").find("geometry").find("box").find("size").text.split()
      pose_x = float(pose[0])
      pose_y = float(pose[1])
      rotation = float(pose[-1])
      if rotation == 0:
        size_x = float(size[0])+margin*2
        size_y = float(size[1])+margin*2
      else:
        size_x = float(size[1])+margin*2
        size_y = float(size[0])+margin*2

      point_1 = [pose_x + size_x / 2, pose_y + size_y / 2]
      point_2 = [point_1[0], point_1[1] - size_y]
      point_3 = [point_1[0] - size_x, point_1[1] - size_y ]
      point_4 = [point_1[0] - size_x, point_1[1] ]

      xmin=min(point_1[0],point_2[0],point_3[0],point_4[0])
      xmax=max(point_1[0],point_2[0],point_3[0],point_4[0])
      ymin=min(point_1[1],point_2[1],point_3[1],point_4[1])
      ymax=max(point_1[1],point_2[1],point_3[1],point_4[1])

      area_points = [xmin, xmax, ymin, ymax]
      obstacle_coordinates.append(area_points)

  return np.array(obstacle_coordinates)
