import gym

def run_env(env_name):
    env = gym.make(env_name)
    env.reset()
    env.render()
    while not is_done:                                          # Loop until game ends
        action = env.action_space.sample()                       
        new_space, reward, is_done, info = env.step(action)
        env.render()
    env.close()

if __name__ == '__main__':
    run_env('SpaceInvaders-v0')
    run_env("LunarLander-v2")
    run_env("CartPole-v0")