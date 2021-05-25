# Uses two different policy functions: target policy(greedy) and behaviour
# policy(stochastic, all probabilities greater than 0) to create the optimal policy.

import torch
import gym
from collections import defaultdict
from tqdm import tqdm
from blackjack_helper import plot_blackjack_values, compare_scores


# Behaviour policy
def generate_random_policy(num_actions):
    probs = torch.ones(num_actions) / num_actions           # Doesn't matter, just send the same probabilities for everything(>0)
    def policy_function(state):
        return probs
    return policy_function


def run_episode(env, behaviour_policy):
    state = env.reset()
    rewards, actions, states = [], [], []
    is_done = False
    while not is_done:
        probs = behaviour_policy(state)
        action = torch.multinomial(probs, 1).item()
        actions.append(action)
        states.append(state)
        state, reward, is_done, info = env.step(action)
        rewards.append(reward)
    return states, actions, rewards


def off_policy_mc_control(env, gamma, num_episodes, behaviour_policy):
    num_actions = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.zeros(num_actions))
    for episode in tqdm(range(num_episodes)):
        weights = {}
        w = 1
        states, actions, rewards = run_episode(env, behaviour_policy)
        return_proba = 0    
        G = {}
        for state, action, reward in zip(reversed(states), reversed(actions), reversed(rewards)):
            return_proba = gamma * return_proba + reward
            G[(state, action)] = return_proba
            w = w / float(behaviour_policy(state)[action])
            weights[(state, action)] = w
            if action != torch.argmax(Q[state]):                   
                # Only continue as long as behaviour and target policies match
                break
           
        for state_action, return_proba in G.items():
            state, action = state_action
            if state[0] <= 21:
                G_sum[state_action] += return_proba * weights[state_action]
                N[state_action] += 1
                Q[state][action] = G_sum[state_action] / N[state_action]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


# Incremental update of Q-function, more efficient and scalable
def off_policy_mc_control_incremental(env, gamma, num_episodes, behaviour_policy):
    num_actions = env.action_space.n
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.zeros(num_actions))
    for episode in tqdm(range(num_episodes)):
        w = 1.0
        states, actions, rewards = run_episode(env, behaviour_policy)
        return_proba = 0    
        for state, action, reward in zip(reversed(states), reversed(actions), reversed(rewards)):
            return_proba = gamma * return_proba + reward
            N[(state, action)] += 1
            Q[state][action] += (w / N[(state, action)]) * (return_proba - Q[state][action])   # Incremental update
            if action != torch.argmax(Q[state]).item():
                break
            w = w / behaviour_policy(state)[action]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


# Weighted importance sampling, which takes the weighted average 
# of returns rather than simple averages. Better the simple averages as variance is much lower.
def off_policy_mc_control_weighted(env, gamma, num_episodes, behaviour_policy):
    num_actions = env.action_space.n
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.zeros(num_actions))
    for episode in tqdm(range(num_episodes)):
        w = 1.0
        states, actions, rewards = run_episode(env, behaviour_policy)
        return_proba = 0    
        for state, action, reward in zip(reversed(states), reversed(actions), reversed(rewards)):
            return_proba = gamma * return_proba + reward
            N[(state, action)] += w                                
            # MAJOR DIFFERENCE IN THIS LINE ONLY: takes weighted counts
            Q[state][action] += (w / N[(state, action)]) * (return_proba - Q[state][action])  
            # Incremental update
            if action != torch.argmax(Q[state]).item():
                break
            w = w / behaviour_policy(state)[action]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    gamma = 1
    epsilon = 0.1
    num_episodes = 50000
    random_policy = generate_random_policy(env.action_space.n)

    optimal_Q, optimal_policy = off_policy_mc_control(env, gamma, num_episodes, random_policy)

    optimal_value = defaultdict(float)
    for state, action_values in optimal_Q.items():
        optimal_value[state] = torch.max(action_values).item()
    plot_blackjack_values(optimal_value)
    compare_scores(env, optimal_policy)

    # Incremental updates
    random_policy = generate_random_policy(env.action_space.n)
    optimal_Q, optimal_policy = off_policy_mc_control_incremental(env, gamma, num_episodes, random_policy)
    compare_scores(env, optimal_policy)

    # Importance sampling
    gamma = 1
    num_episodes = 50000
    random_policy = generate_random_policy(env.action_space.n)

    optimal_Q, optimal_policy = off_policy_mc_control_incremental(env, gamma, num_episodes, random_policy)

    compare_scores(env, optimal_policy)