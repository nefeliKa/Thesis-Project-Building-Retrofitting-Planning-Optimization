import numpy as np
import gym
from gym.spaces import Discrete

# - House with energy demand 190 kwh/year
# - house consists of: roof, wall, cellar
# roof --> 57
# wall --> 95
# cell --> 38

# - house energy demand : D_roof + D_wall + D_cell
# - Material performance classes
# [90 - 100] % ---> s1
# [70 - 89 ] % ---> s2
# [0  - 69 ] % ---> s3

# Time horizon: 50 yrs
# Time step: 5 yrs

# Renovation costs
# roof --> 3000 E
# wall --> 2000 E
# cell --> 5000 E


class House(gym.Env):
    def __init__(self):
        self.house_size_m2 = 120    # m^2
        self.action_space = Discrete(4)
        # 0 : Do nothing
        # 1 : Roof renovation
        # 2 : Wall renovation
        # 3 : Cell renovation
        self.components = {
            "roof": 0,
            "wall": 1,
            "cellar": 2
        }
        self.material_perf = ["good", "medium", "bad"]

        self.years = ['year_0', 'year_5', 'year_10', 'year_15', 'year_20', 'year_25', 'year_30', 'year_35', 'year_40', 'year_45', 'year_50']
        self.observation_space = [(r, w, c, year) for r in self.material_perf
                                                  for w in self.material_perf
                                                  for c in self.material_perf
                                                  for year in self.years]
        self.state = ('good', 'good', 'good', 'year_0')
        self.num_years = 50

        self.renovation_costs = np.array([0, 3000, 2000, 5000])  # Cost of renovating each component

        # Energy demand for Roof, Wall, Cellar. We start from nominal state
        self.energy_demand_nominal = [57, 95, 38]   # Initial energy demand for "roof", "wall", "cellar"
        self.degradation_rates = [0.0, 0.2, 0.4]    # degradation rates for "good", "medium", "bad" states

        self.transition_matrix_renovate = [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]

        self.transition_matrix_do_nothing = [
            [0.8, 0.2, 0.0],
            [0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0],
        ]

        energy_demand_roof = self.energy_demand_nominal[self.components["roof"]]
        energy_demand_wall = self.energy_demand_nominal[self.components["wall"]]
        energy_demand_cell = self.energy_demand_nominal[self.components["cellar"]]
        self.energy_bills_prev = (self.energy2euros(energy_demand_roof) + self.energy2euros(energy_demand_wall) +
                                  self.energy2euros(energy_demand_cell)) * self.house_size_m2

        self.current_time_step = 0

    @staticmethod
    def energy2euros(num_of_kwh: float) -> float:
        price_kwh = 0.35  # currently in the NL
        return price_kwh * num_of_kwh

    def step(self, action):
        # Find next state
        transition_matrix = self.transition_matrix_renovate
        if action == 0:
            transition_matrix = self.transition_matrix_do_nothing

        (roof_health, wall_health, cellar_health, year) = self.state

        index_roof = self.material_perf.index(roof_health)
        index_wall = self.material_perf.index(wall_health)
        index_cellar = self.material_perf.index(cellar_health)

        roof_health_new = np.random.choice(len(self.material_perf), p=transition_matrix[index_roof])
        wall_health_new = np.random.choice(len(self.material_perf), p=transition_matrix[index_wall])
        cell_health_new = np.random.choice(len(self.material_perf), p=transition_matrix[index_cellar])

        state_new = (self.material_perf[roof_health_new],
                     self.material_perf[wall_health_new],
                     self.material_perf[cell_health_new],
                     self.years[self.years.index(year) + 1])
        print(f"OLD STATE: {self.state}")
        print(f"NEW STATE: {state_new}")

        # calculate reward

        # reward = renovation_costs + energy_bills
        renovation_costs = self.renovation_costs[action]

        # kWh
        energy_demand_roof = self.energy_demand_nominal[self.components["roof"]] * (1 + self.degradation_rates[roof_health_new])
        energy_demand_wall = self.energy_demand_nominal[self.components["wall"]] * (1 + self.degradation_rates[wall_health_new])
        energy_demand_cell = self.energy_demand_nominal[self.components["cellar"]] * (1 + self.degradation_rates[cell_health_new])
        total_energy_demand = energy_demand_roof + energy_demand_wall + energy_demand_wall

        # kWh -> Euros
        energy_bills = (self.energy2euros(energy_demand_roof) +
                        self.energy2euros(energy_demand_wall) +
                        self.energy2euros(energy_demand_cell))
        energy_bills = energy_bills * self.house_size_m2
        # energy_costs_diff = energy_bills - self.energy_bills_prev

        net_cost = renovation_costs + energy_bills
        reward = -net_cost
        print(f"     Energy demand     : Roof: {energy_demand_roof:.2f}, Wall: {energy_demand_wall:.2f}, Cell: {energy_demand_cell:.2f}")
        print(f"     Net cost          : {net_cost:.2f} Euros")
        print(f"        Renovation         : {renovation_costs:.2f} Euros")
        print(f"        new_energy_bills   : {energy_bills:.2f} Euros")
        print(f"        old_energy_bills   : {self.energy_bills_prev:.2f} Euros")

        print(f"     Reward            : {reward:.2f} Euros")

        self.energy_bills_prev = energy_bills

        # is terminal
        self.current_time_step += 1

        is_terminal = (self.current_time_step >= 10) or (total_energy_demand > 250)

        self.state = state_new
        return self.state, reward, is_terminal, {}

    def reset(self):
        self.state = ('good', 'good', 'good', 'year_0')
        self.current_time_step = 0
        energy_demand_roof = self.energy_demand_nominal[self.components["roof"]]
        energy_demand_wall = self.energy_demand_nominal[self.components["wall"]]
        energy_demand_cell = self.energy_demand_nominal[self.components["cellar"]]
        self.energy_bills_prev = (self.energy2euros(energy_demand_roof) + self.energy2euros(energy_demand_wall) +
                                  self.energy2euros(energy_demand_cell)) * self.house_size_m2

        return self.state


def main():
    env = House()
    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            # env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))
        print()


if __name__ == "__main__":
    main()
