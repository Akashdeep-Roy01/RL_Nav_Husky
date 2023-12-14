import os
import rclpy
import optuna
from Husky_SB3env import *
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

def main():

    rclpy.init()
    # Hyperparameter tuning using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=30, n_jobs=1, show_progress_bar = True)
    # Print best params
    print("Best Hyperparameters: " + str(study.best_params))
    rclpy.shutdown()


def optimize_sac(trial):
    ## This method defines the range of hyperparams to search fo the best tuning
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3),
        'batch_size':trial.suggest_int('batch_size',256,1024,step=256),
        'tau':trial.suggest_float('tau',0,1),
        'gamma': trial.suggest_float('gamma', 0.8, 0.9999), 
        'target_update_interval': trial.suggest_int('policy_delay', 1, 50),         
    }


def optimize_agent(trial):
    ## This method is used to optimize the hyperparams for our problem
    try:
        # Create environment
        env_opt = Husky_SB3Env()
        env_opt = Monitor(env_opt)
        check_env(env_opt)
        # Setup dirs
        dir = 'runs'
        log_dir = os.path.join(dir, 'logs')
        SAVE_PATH = os.path.join(log_dir, 'tuning', 'trial_{}'.format(trial.number))
        # Setup the parameters
        model_params = optimize_sac(trial)
        # Setup the model
        n_actions = env_opt.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = SAC("MultiInputPolicy", env_opt, action_noise=action_noise, \
                    ent_coef="auto", target_entropy="auto",\
                    tensorboard_log=log_dir, verbose=0, **model_params)
        model.set_random_seed(123456)
        model.learn(total_timesteps=100000)
        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env_opt, n_eval_episodes=20)
        # Close env and delete
        env_opt.close()
        del env_opt

        model.save(SAVE_PATH)
        model.save_replay_buffer(f"{SAVE_PATH}/SAC_buffer")

        return mean_reward

    except Exception as e:
        return -10000
    
if __name__ == "__main__":
    main()