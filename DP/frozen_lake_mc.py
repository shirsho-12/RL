# Value estimation for the FrozenLake problem with first-visit and every-visit 
# Monte Carlo policy evaluation. 

import torch
import gym


def run_episode(env, policy):
    state = env.reset()
    states, rewards, is_done = [state], [], False
    while not is_done:
        action = policy[state].item()
        state, reward, is_done, info = env.step(action)
        # All states and rewards recorded as the full environment is not accessible
        states.append(state)
        rewards.append(reward)  
    states = torch.tensor(states)
    rewards = torch.tensor(rewards)
    return states, rewards


def mc_prediction_first_visit(env, policy, gamma, num_episodes):
    num_states = policy.shape[0]
    V, N = torch.zeros(num_states), torch.zeros(num_states)
    for episode in range(num_episodes):
        states, rewards = run_episode(env, policy)
        first_visit, G = torch.zeros(num_states), torch.zeros(num_states)
        return_prob = 0
        # Skip initial state
        for state, reward in zip(reversed(states[1:]), reversed(rewards)):
             # First return (put in reverse otherwise rewards start at 0)
            return_prob = gamma * return_prob + reward            
            G[state] = return_prob
            first_visit[state] = 1   
        for state in states:
            if first_visit[state]:
                V[state] += G[state]
                N[state] += 1
    for state in range(num_states):
        if N[state] > 0:
            V[state] = V[state] / N[state]     # Averaging first returns
    return V


def mc_prediction_every_visit(env, policy, gamma, num_episodes):
    num_states = policy.shape[0]
    V, N = torch.zeros(num_states), torch.zeros(num_states)
    G = torch.zeros(num_states)
    for episode in range(num_episodes):
        states, rewards = run_episode(env, policy)
        return_prob = 0
        # Skip initial state
        for state, reward in zip(reversed(states[1:]), reversed(rewards)):
             # Every return added
            return_prob = gamma * return_prob + reward            
            G[state] += return_prob
            N[state] += 1
    for state in range(num_states):
        if N[state] > 0:
            V[state] = G[state] / N[state]     # Averaging total returns
    return V


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    gamma = 1
    num_episodes = 1000
    # Optimal policy from DP solution
    optimal_policy = torch.tensor([0., 3., 3., 3., 0., 3., 2., 3., 3., 1., 0., 3., 3., 2., 1., 3.])
    mc_optimal_values = mc_prediction_first_visit(env, optimal_policy, gamma, num_episodes)
    print("First visit MC values: \n", mc_optimal_values)
    mc_optimal_values = mc_prediction_every_visit(env, optimal_policy, gamma, num_episodes)
    print("Every visit MC values: \n", mc_optimal_values)