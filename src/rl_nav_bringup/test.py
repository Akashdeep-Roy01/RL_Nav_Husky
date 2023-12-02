from Husky_SB3env import *
from stable_baselines3.common.env_checker import check_env

rclpy.init()
env = Husky_SB3Env()
check_env(env)
