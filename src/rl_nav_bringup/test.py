import rclpy
import numpy as np
from Husky_SB3env import *
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

def main():

    rclpy.init()
    env = Husky_SB3Env()
    env = Monitor(env)
    check_env(env)
    trained_model_path = "/docker_ws/local/rl_nav_bringup/backup/best_model_3.zip"
    custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}
    model = SAC.load(trained_model_path, env=env, custom_objects=custom_obj)
    Mean_ep_rew, Num_steps = evaluate_policy(model, env=env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True)
    env.unwrapped.get_logger().info("Mean_reward: "+str(np.mean(Mean_ep_rew)))
    env.unwrapped.get_logger().info("Mean_episode_length: "+str(np.mean(Num_steps)))
    env.close()
    del env

if __name__=="__main__":
    main()