# Solving the FrozenLake problem with policy iteration: https:/​/​gym.​openai.​com/​envs/FrozenLake-​v0/
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


def policy_evaluation(env, policy, gamma, threshold):
    num_states = env.observation_space.n
    V = torch.zeros(num_states)
    max_delta = threshold + 1
    while max_delta > threshold:
        temp = torch.zeros(num_states)
        for state in range(num_states):
            action = policy[state].item()
            for proba, new_state, reward, _ in env.env.P[state][action]:
                temp[state] += proba * (reward + V[new_state] * gamma)
        max_delta = torch.max(torch.abs(V - temp))
        V = temp.clone()
    return V


def policy_improvement(env, V, gamma):
    num_states, num_actions = env.observation_space.n, env.action_space.n
    policy = torch.zeros(num_states)
    for state in range(num_states):
        actions = torch.zeros(num_actions)
        for action in range(num_actions):
            for proba, new_state, reward, _ in env.env.P[state][action]:
                actions[action] += proba * (reward + V[new_state] * gamma)
        policy[state] = torch.argmax(actions)
    return policy


def policy_iteration(env, gamma=0.99, threshold=0.0001):
    num_states, num_actions = env.observation_space.n, env.action_space.n
    policy = torch.randint(high=num_actions, size=(num_states,)).float()
    while True:
        V = policy_evaluation(env, policy, gamma=gamma, threshold=threshold)
        new_policy = policy_improvement(env, V, gamma=gamma)
        if torch.equal(new_policy, policy):
            return V, new_policy
        policy = new_policy.clone()


if __name__ == "__main__":
    NUM_EPISODES = 1000
    env = gym.make("FrozenLake-v0") 
    # env = gym.make("FrozenLake8x8-v0")
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(num_states, num_actions)
    V_optimal, optimal_policy = policy_iteration(env)
    print("Optimal values:\n",  V_optimal)
    print("Optimal policy", optimal_policy)

    total_reward = []
    for n in range(NUM_EPISODES):
        total_reward.append(run_episode(env, optimal_policy))
    print(f"Success rate over {NUM_EPISODES} episodes: {sum(total_reward) * 100 / NUM_EPISODES}%")