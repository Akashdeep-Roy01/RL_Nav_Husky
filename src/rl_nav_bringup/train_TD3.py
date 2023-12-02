import os
import numpy as np
from Husky_SB3env import *
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback

def main(mode="training"):

    rclpy.init()
    # Create the dir where the trained RL models will be saved
    dir = 'runs'
    trained_models_dir = os.path.join(dir, 'rl_models')
    log_dir = os.path.join(dir, 'logs')
    
    # If the directories do not exist we create them
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = Husky_SB3Env()
    env = Monitor(env)
    check_env(env)

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    eval_callback = EvalCallback(env, best_model_save_path=trained_models_dir,
                             log_path=log_dir, eval_freq=10000,
                             deterministic=True, render=False)

    if mode == "training":  
        model = TD3("MultiInputPolicy", env, learning_starts = 10000, buffer_size = 500000, \
                    batch_size = 512, action_noise=action_noise,\
                    tensorboard_log=log_dir, verbose=1)
        model.set_random_seed(123456)
        try:    
            model.learn(total_timesteps=1000000, log_interval=50, progress_bar=True,callback=eval_callback)
        except KeyboardInterrupt:
            model.save(f"{trained_models_dir}/TD3_model")
            model.save_replay_buffer(f"{trained_models_dir}/TD3_buffer")
        model.save(f"{trained_models_dir}/TD3_model")

    elif mode == "retraining":
        trained_model_path = os.path.join(dir, 'rl_models', 'TD3_model.zip') #Change path
        replay_buffer_path = os.path.join(dir, 'rl_models', 'TD3_buffer.pkl') #Change path
        custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}
        model = TD3.load(trained_model_path, env=env, custom_objects=custom_obj)
        model.load_replay_buffer(replay_buffer_path)
        model.set_random_seed(123456)
        # Execute training
        try:
            print("success")
            model.learn(total_timesteps=int(1000000), log_interval=50, progress_bar=True,callback=eval_callback)
        except KeyboardInterrupt:
            model.save(f"{trained_models_dir}/TD3_model")
            model.save_replay_buffer(f"{trained_models_dir}/TD3_buffer")
        model.save(f"{trained_models_dir}/TD3_model")


    rclpy.shutdown()

if __name__ == "__main__":
    main()