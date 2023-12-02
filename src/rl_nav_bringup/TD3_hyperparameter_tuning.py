import os
import rclpy
import optuna
from Husky_SB3env import *
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

def main():

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
    env.close()
    del env
    # Hyperparameter tuning using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=10, n_jobs=1)
    # Print best params
    print("Best Hyperparameters: " + str(study.best_params))

    rclpy.shutdown()


def optimize_td3(trial):
    ## This method defines the range of hyperparams to search fo the best tuning
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3),
        'gamma': trial.suggest_float('gamma', 0.8, 0.9999), 
        'policy_delay': trial.suggest_int('policy_delay', 1, 50), 
        'tau':trial.suggest_float('tau',0,1),
        'target_policy_noise': trial.suggest_float('target_policy_noise', 0, 1),
        'target_noise_clip': trial.suggest_float('target_noise_clip', 0, 1),
        'batch_size':trial.suggest_int('batch_size',256,1024,step=256)
    }


def optimize_agent(trial):
    ## This method is used to optimize the hyperparams for our problem
    try:
        # Create environment
        env_opt = Husky_SB3Env()
        env_opt = Monitor(env_opt)
        # Setup dirs
        dir = 'runs'
        log_dir = os.path.join(dir, 'logs')
        SAVE_PATH = os.path.join(log_dir, 'tuning', 'trial_{}'.format(trial.number))
        # Setup the parameters
        model_params = optimize_td3(trial)
        # Setup the model
        model = TD3("MultiInputPolicy", env_opt, tensorboard_log=log_dir, verbose=0, **model_params)
        model.learn(total_timesteps=150000)
        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env_opt, n_eval_episodes=20)
        # Close env and delete
        env_opt.close()
        del env_opt

        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:
        return -10000
    
if __name__ == "__main__":
    main()