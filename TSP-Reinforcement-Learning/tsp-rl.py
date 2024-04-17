from TSP.envs.tsp_env import TSPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
import gymnasium
import numpy as np


def main():
    # !!! check TSP/envs/tsp_env.py for the traveling salesman enviroment I made that this runs off of

    graph = np.array([[0, 16, 11, 6],
                    [8, 0, 13, 16],
                    [4, 7, 0, 9],
                    [5, 12, 2, 0]])
    names = {
        0: "A",
        1: "B",
        2: "C",
        3: "D"
    }
    env = make_vec_env('TSP-v0', n_envs=30, env_kwargs={'graph': graph, 'names': names})

    train = False
    if train:
        model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=0.0003)
        for i in range(10):
            model.learn(total_timesteps=100000)
            model.save(f"ppo_tsp_{i+1}")
    else:
        model = PPO.load("ppo_tsp")

    env = TSPEnv(graph, names)
    obs, info = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, truncated, done, info = env.step(action)
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(i + 1))
            break

if __name__ == "__main__":
    main()