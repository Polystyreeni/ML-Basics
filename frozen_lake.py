"""
Reinforced learning exercise: Frozen lake
Excercies uses the OpenAI Gym environment

The goal of the excerice is to learn the optimal policy to solve the
Frozen lake problem, using Q-learning algorithm

For more info on Frozen lake:
https://www.gymlibrary.dev/environments/toy_text/frozen_lake/

"""


import gym
import numpy as np
import matplotlib.pyplot as plt


# Evaluate the current Q-table
def eval_policy(qtable_, num_of_episodes_, max_steps_, env_):
    rewards = []
    for episode in range(num_of_episodes_):
        state = env_.reset()
        step = 0
        done = False
        total_rewards = 0
        for step in range(max_steps_):
            action = np.argmax(qtable_[state, :])
            new_state, reward, done, info = env_.step(action)
            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    avg_reward = sum(rewards) / num_of_episodes_
    return avg_reward


# Returns Q-table and rewards for a single learning run
def q_learn(env_, episodes_, steps_, alpha_, gamma_, non_deterministic=False):
    qtable = np.zeros((16, 4))  # Initialize Q-table
    rewards = []
    for N in range(episodes_):  # Go through N episodes
        state = env_.reset()
        done = False

        for M in range(steps_):
            action = np.random.randint(0, 4)
            new_state, reward, done, info = env_.step(action)

            if non_deterministic:
                qtable[state, action] = qtable[state, action] + alpha_ \
                                        * (reward + gamma_ * np.max(qtable[new_state]) - qtable[state, action])
            else:
                qtable[state, action] = reward + gamma_ * np.max(qtable[new_state])

            if done:
                break
            state = new_state

        rewards.append(eval_policy(qtable, episodes_, steps_, env_))

    return qtable, rewards


# Q-learning algorithm
episodes = 200
steps = 100
alpha = 0.5
gamma = 0.9
runs = 10

print("Deterministic case, deterministic update rule...")
env_det = gym.make("FrozenLake-v1", is_slippery=False)
env_det.reset()
det_rewards = []

for run in range(runs):
    qtable, rewards = q_learn(env_det, episodes, steps, alpha, gamma)
    det_rewards.append(rewards)

print("Deterministic learning complete!")
print()

print("Non-deterministic case, deterministic update rule...")

env_slippery = gym.make("FrozenLake-v1", is_slippery=True)
env_slippery.reset()
non_det_rewards = []
for run in range(runs):
    qtable, rewards = q_learn(env_slippery, episodes, steps, alpha, gamma)
    non_det_rewards.append(rewards)

print("Non-deterministic learning complete!")
print()

print("Non deterministic case, non-deterministic update rule...")
non_det_up_rewards = []
env_slippery.reset()
for run in range(runs):
    qtable, rewards = q_learn(env_slippery, episodes, steps, alpha, gamma, True)
    non_det_up_rewards.append(rewards)

print("Non-deterministic learning with update rule complete!")
print()

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.tight_layout(pad=1.0)

ax1.set_title("Deterministic case")
ax2.set_title("Non-deterministic case, normal update rule")
ax3.set_title("Non-deterministic case, non-deterministic update rule")

ax1.set(xlabel='Episodes', ylabel='Average reward')
ax2.set(xlabel='Episodes', ylabel='Average reward')
ax3.set(xlabel='Episodes', ylabel='Average reward')

# Plot graph for each run
x = list(range(0, episodes))
for i in range(runs):
    y1 = det_rewards[i]
    y2 = non_det_rewards[i]
    y3 = non_det_up_rewards[i]

    label = f"Run {i + 1}"
    ax1.plot(x, y1, label=label)
    ax2.plot(x, y2, label=label)
    ax3.plot(x, y3, label=label)

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')

plt.show()


