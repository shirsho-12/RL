import torch
import gym
from collections import defaultdict
from tqdm import tqdm
from blackjack_helper import plot_blackjack_values, compare_scores


def run_episode(env, Q, num_actions):
    """
    env: OpenAI Gym environment
    Q: Action-value maatrix, a.k.a. Q-function
    num_actions: action space
    returns states, actions and rewards for the episode
    """
    state = env.reset()
    states, actions, rewards = [], [], []
    is_done = False
    action = torch.randint(low=0,high=num_actions, size=[1]).item()        # EXPLORING STARTS
    while not is_done:
        actions.append(action)
        states.append(state)
        state, reward, is_done, info = env.step(action)
        rewards.append(reward)
        action = torch.argmax(Q[state]).item()
    return states, actions, rewards


def mc_control_on_policy(env, gamma, num_episodes):
    num_actions = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.zeros(num_actions))
    for episode in tqdm(range(num_episodes)):
        states, actions, rewards = run_episode(env, Q, num_actions)
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
            # Making the policy greedy
            policy[state] = torch.argmax(actions).item()
    return Q, policy


if __name__ == "__main__":
    gamma = 1
    num_episodes = 50000
    env = gym.make("Blackjack-v0")
    optimal_Q, optimal_policy = mc_control_on_policy(env, gamma, num_episodes)

    optimal_value = defaultdict(float)
    for state, action_values in optimal_Q.items():
        optimal_value[state] = torch.max(action_values).item()
    plot_blackjack_values(optimal_value)
    compare_scores(env, optimal_policy)
