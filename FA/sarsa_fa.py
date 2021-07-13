from tqdm import tqdm
from q_learning_fa import gen_eps_greedy_policy

def sarsa(env, estimator, num_episodes, gamma=1.0, epsilon=0.1, eps_decay=0.99):
    # Don't really see a point in adding a decay, but might help -> LambdaLR better
    total_reward_episode = [0] * num_episodes # For graphing
    for episode in tqdm(range(num_episodes)):
        policy = gen_eps_greedy_policy(estimator, epsilon, env.action_space.n)
        state = env.reset()
        action = policy(state)
        is_done = False

        while not is_done:
            next_state, reward, is_done, _ = env.step(action)
            next_q_values = estimator.predict(next_state)
            next_action = policy(next_state)
            td_target = reward + gamma * next_q_values[next_action]
            estimator.update(state, action, td_target)
            total_reward_episode[episode] += reward
            state = next_state
            action = next_action            
    epsilon *= eps_decay

    return total_reward_episode\
