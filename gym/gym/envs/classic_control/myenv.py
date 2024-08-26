from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np 
from gym.envs.registration import register
from stable_baselines3.common.env_checker import check_env

class MyEnv(Env):
    def __init__(self):
        self.num_components = 3
        self.max_time = 100 #max time of env
        self.action_space = spaces.MultiDiscrete([4, 4, 4])
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(3,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set seed if provided

        self.state = np.array([0, 0, 0])
        self.time_step = 0
        return self.state, None


    def step(self, actions): 

        state = self.state
        next_states = []
        reward = 0
        for comp, action in zip(state,actions):
            if any(action in actions == 0):
                next_state = comp + 10
                reward += 0
            elif action == 1:
                next_state = comp
                reward += 100
            elif action == 2:
                next_state = max(0, comp - 5)
                reward += 200
            else:
                next_state = 0
                reward += 500
            next_states.append(next_state)

        fail_cost = any(value >= 30 for value in self.state) * 800
        reward -= fail_cost

        self.state = next_states

        done = True

        if self.time_step == 100:
            done = True
        else: 
            self.time_step = self.time_step + 10

        observation = np.array(self.state, dtype=np.int32)  # Convert state to numpy array with dtype int32
        # observation = observation.reshape(self.observation_space.shape)  # Ensure observation has the correct shape

        info = {}
        return observation, reward, done, False ,info

    def render(): 
        pass


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env =  make_vec_env(MyEnv)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000)
model.save("TempleENv")

del model # remove to demonstrate saving and loading

model = PPO.load("TempleENv")

obs = env.reset()
while True:
    actions = model.predict(obs)
    print("Action:", actions)
    obs, rewards, dones, info = env.step(actions)
    print("Observation:", obs)
    print("Rewards:", rewards)
    print("Dones:", dones)
    print("Info:", info)
    env.render()
