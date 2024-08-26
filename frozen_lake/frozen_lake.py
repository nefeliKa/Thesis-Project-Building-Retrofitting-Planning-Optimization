import gym

env = gym.make("FrozenLake-v0")
env.render()

##### STATES
# encode all (16) discrete states: S --> 0, F --> 1, ... G --> 15
# print(env.observation_space)

#### ACTIONS
# Encode all possible actions
# 0 --> L
# 1 --> D
# 2 --> R
# 3 --> U
# print(env.action_space)

### TRANSITION PROB
# We learned that state S is
# encoded as 0 and the action right is encoded as 2, so, to obtain the
# transition probability of state S by performing the action right, we type
# env.P[0][2] as the following shows:
# print(env.P[0][2])

#   We reach state 4 (F) with probability 0.33333 and receive 0 reward.
#   We reach state 1 (F) with probability 0.33333 and receive 0 reward.
#   We reach the same state 0 (S) with probability 0.33333 and receive 0 reward.

# Thus, when we type env.P[state][action], we get the result in the form
# of [(transition probability, next state, reward, Is terminal
# state?)]

############################### Generating an episode
### Action Selection
env.reset()
# env.step(1)     # Encoded as "I want to go down"
(next_state, reward, done, info) = env.step(1)
print((next_state, reward, done, info))
env.render()

# Make a random action instead of a predefined one (like e.g. action 1 previously)
random_action = env.action_space.sample()
next_state, reward, done, info = env.step(random_action)
print((next_state, reward, done, info))
env.render()

### Generate an episode

print("#################### Many episodes ###########################")
# Random policy
num_episodes = 10
num_timesteps = 20
for i in range(num_episodes):
    print(f"########### Episode: {i}")
    env.reset()
    print('Time Step 0: ')
    env.render()
    for t in range(num_timesteps):
        random_action = env.action_space.sample()
        next_state, reward, done, info = env.step(random_action)
        print('Time Step {} :'.format(t + 1))
        env.render()
        if done:
            break
