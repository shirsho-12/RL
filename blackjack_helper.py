import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from tqdm import tqdm

def plot_surface(x, y, z, title):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, 
                              vmin=-1.0, vmax=1.0)
    ax.set_xlabel("Player Sum")
    ax.set_ylabel("Dealer showing")
    ax.set_zlabel("Value")
    ax.set_title(title)
    ax.view_init(ax.elev, 110)          # Set camera elevation
    fig.colorbar(surface)
    plt.show()


def plot_blackjack_values(V):
    player_sum = range(12, 22)
    dealer_values = range(1, 11)
    x, y = torch.meshgrid(torch.tensor(player_sum), torch.tensor(dealer_values))
    plot_values = torch.zeros((len(player_sum), len(dealer_values), 2))
    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_values):
            for k, ace in enumerate([False, True]):
                plot_values[i, j, k] = V[player, dealer, ace]
    plot_values = plot_values.numpy()
    plot_surface(x, y, plot_values[:, :, 0], "Blackjack value function without a usable ace")
    plot_surface(x, y, plot_values[:, :, 1], "Blackjack value function with a usable ace")


def simulate_episode(env, policy):
    state = env.reset()
    is_done = False
    while not is_done:
        action = policy[state]
        state, reward, is_done, info = env.step(action)
    return reward


def compare_scores(env, optimal_policy):
    hold_score = 18
    hold_policy = {}
    for player in range(2, 22):
        for dealer in range(1, 11):
            action = 1 if player < hold_score else 0
            for ace in [True, False]:
                hold_policy[(player, dealer, ace)] = action

    num_episodes = 50000
    optimal_scores = [0, 0]
    simple_scores = [0, 0]
    for episode in tqdm(range(num_episodes)):
        reward = simulate_episode(env, optimal_policy)
        if reward == 1:
            optimal_scores[0] += 1
        elif reward == -1:
            optimal_scores[1] += 1
        reward = simulate_episode(env, hold_policy)
        if reward == 1:
            simple_scores[0] += 1
        elif reward == -1:
            simple_scores[1] += 1
    print(f"\nOptimal policy:\nWin rate: {optimal_scores[0] / num_episodes * 100 :.2f}%" 
    f"\tLoss rate: {optimal_scores[1] / num_episodes * 100:.2f}%")
    print(f"Simple policy:\nWin rate: {simple_scores[0] / num_episodes * 100:.2f}%" 
    f"\tLoss rate: {simple_scores[1] / num_episodes * 100:.2f}%")