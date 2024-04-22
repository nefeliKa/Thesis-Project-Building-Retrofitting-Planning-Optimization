from house_environment import House
from stable_baselines3.common.env_checker import check_env

env = House()
check_env(env)