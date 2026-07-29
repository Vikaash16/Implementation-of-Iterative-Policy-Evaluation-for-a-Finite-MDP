import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# Access the unwrapped environment to use the transition model
env = env.unwrapped

# Number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Parameters
gamma = 0.99
theta = 1e-8

# Random policy: each action has equal probability
policy = np.ones((n_states, n_actions)) / n_actions

# Initialize value function
V = np.zeros(n_states)

# -------------------------------------------------
# Policy Evaluation Function
# -------------------------------------------------

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Performs iterative policy evaluation using the Bellman expectation equation.

    Parameters:
        env    : Gymnasium FrozenLake environment
        policy : Fixed policy to be evaluated
        gamma  : Discount factor
        theta  : Convergence threshold

    Returns:
        V         : Estimated state-value function
        iteration : Number of iterations used for convergence
    """

    V = np.zeros(n_states)
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):
            v = V[s]
            new_v = 0

            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    new_v += policy[s][a] * prob * (reward + gamma * V[next_state])

            V[s] = new_v
            delta = max(delta, abs(v - new_v))

        iteration += 1

        if delta < theta:
            break

    return V, iteration

# Run policy evaluation
V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name:Vikaash P")
print("Register Number:212223240180")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("Name:Vikaash P")
print("Register Number:212223240180    ")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()
