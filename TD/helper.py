import torch
import matplotlib.pyplot as plt 
from gym.wrappers import Monitor

def run_episode(env, policy):
    # Visual representation of actions taking place, saved in video directory
    # Requires ffmpeg to be installed to run
    env = Monitor(env, './video', force=True)
    is_done = False
    total_reward = 0
    state = env.reset()
    while not is_done:
        action = policy[state]
        state, reward, is_done, info = env.step(action)
        total_reward += reward
    env.close()
    print("Total reward:", total_reward)


def gen_eps_greedy_policy(num_actions, epsilon):
    # Epsilon Greedy exploratory policy 
    def policy_function(state, Q):
        probs = torch.ones(num_actions) * epsilon / num_actions
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


def plot_rate(episode_length, total_reward_episode, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(episode_length)
    ax[0].set_title("Episode Length over time")
    ax[0].set(xlabel="Episode", ylabel="Length")
    ax[1].plot(total_reward_episode)
    ax[1].set_title("Episode reward over time")
    ax[1].set(xlabel="Episode reward over time", ylabel="Reward")
    fig.suptitle(title)

    plt.show()