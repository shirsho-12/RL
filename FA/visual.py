import matplotlib.pyplot as plt


def reward_visual(total_reward, title="Episode reward over time"):
    plt.plot(total_reward)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
