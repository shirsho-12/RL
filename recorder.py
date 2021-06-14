import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder


video_dir = './video/pacman.mp4'

env = gym.make("MsPacman-v0")
# env = gym.wrappers.Monitor(env, video_dir)
video_recorder = VideoRecorder(env, video_dir, enabled=True)

is_done = False
env.reset()
while not is_done:        # Loop until game ends
    env.render()
    video_recorder.capture_frame()                                  
    action = env.action_space.sample()
    new_space, reward, is_done, info = env.step(action)

video_recorder.close()
video_recorder.enabled = False
env.close()
