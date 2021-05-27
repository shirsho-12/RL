# Solving the FrozenLake problem with value iteration: https:/​/​gym.​openai.​com/​envs/FrozenLake-​v0/
import torch
import gym


def run_episode(env, policy):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward, is_done, info = env.step(action)
        total_reward += reward
    return total_reward


def value_iteration(env, gamma=0.99, thres = 0.0001):
    num_states, num_actions = env.observation_space.n, env.action_space.n
    V = torch.zeros(num_states)
    max_delta = thres + 1
    while max_delta > thres:
        temp = torch.zeros(num_states)
        for state in range(num_states):
            v_actions = torch.zeros(num_actions)
            for action in range(num_actions):
                for proba, new_state, reward, is_done in env.env.P[state][action]:
                    v_actions[action] += proba * (reward + gamma * V[new_state])    # Value iteration 
            temp[state] = torch.max(v_actions)              # Select the action with the highest reward
        max_delta = torch.max(torch.abs(V - temp))
        V = temp.clone()
    return V


def extract_optimal_policy(env, V, gamma=0.99):
    num_states, num_actions = env.observation_space.n, env.action_space.n
    optimal_policy = torch.zeros(num_states)
    for state in range(num_states):
        v_actions = torch.zeros(num_actions)
        for action in range(num_actions):
            for proba, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += proba * (reward + gamma * V[new_state])
        optimal_policy[state] = torch.argmax(v_actions)
    return optimal_policy

if __name__ == "__main__":
    NUM_EPISODES = 1000
    env = gym.make("FrozenLake-v0") 
    # env = gym.make("FrozenLake8x8-v0")
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(num_states, num_actions)

    optimal_values = value_iteration(env)
    print("Optimal values:\n",  optimal_values)
    optimal_policy = extract_optimal_policy(env, optimal_values)
    print("Optimal policy", optimal_policy)

    total_reward = []
    for n in range(NUM_EPISODES):
        total_reward.append(run_episode(env, optimal_policy))
    print(f"Success rate over {NUM_EPISODES} episodes: {sum(total_reward) * 100 / NUM_EPISODES}%")