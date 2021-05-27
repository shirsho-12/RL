import torch
import gym
from collections import defaultdict
from tqdm import tqdm
from MC.blackjack_helper import plot_blackjack_values, compare_scores


def run_eps_episode(env, Q, epsilon, num_actions):
    state = env.reset()
    rewards, actions, states = [], [], []
    is_done = False
    while not is_done:
        proba = torch.ones(num_actions) * epsilon / num_actions    
        # epsilon-greedy starting probabilities
        best_action = torch.argmax(Q[state]).item()
        proba[best_action] += 1 - epsilon
        action = torch.multinomial(proba, 1).item()                 
        # Choose 1 action from the multinomial distribution
        actions.append(action)
        states.append(state)
        state, reward, is_done, info = env.step(action)
        rewards.append(reward)
    return states, actions, rewards


def mc_control_eps_greedy(env, gamma, num_episodes, epsilon):
    num_actions = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.zeros(num_actions))
    for episode in tqdm(range(num_episodes)):
        states, actions, rewards = run_eps_episode(env, Q, epsilon, num_actions)          
        # only difference in this function 
        return_proba = 0
        G = {}
        for state, action, reward in zip(states[::-1], actions[::-1], rewards[::-1]):        
            # First visit MC policy value prediction
            return_proba = gamma * return_proba + reward
            G[(state, action)] = return_proba
        for (state, action), return_proba in G.items():                                      
            # Update Q function
            if state[0] <= 21:
                G_sum[(state, action)] += return_proba
                N[(state, action)] += 1
                Q[state][action] = G_sum[(state, action)] / N[(state, action)]
        policy = {}
        for state, actions in Q.items():                                                     
            # Making the policy greedy (less so)
            policy[state] = torch.argmax(actions).item()
    return Q, policy


if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    gamma = 1
    epsilon = 0.1
    num_episodes = 50000
    optimal_Q, optimal_policy = mc_control_eps_greedy(env, gamma, num_episodes, epsilon)

    optimal_value = defaultdict(float)
    for state, action_values in optimal_Q.items():
        optimal_value[state] = torch.max(action_values).item()
    plot_blackjack_values(optimal_value)
    compare_scores(env, optimal_policy)