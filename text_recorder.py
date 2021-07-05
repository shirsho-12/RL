# Using string_recorder https://github.com/kiyukuta/string_recorder.git for text-based environments
import subprocess
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import string_recorder

def record(env, policy, video_dir):
    rec = string_recorder.StringRecorder()  
    timestep_limit = 30

    video_recorder = VideoRecorder(env, video_dir, enabled=True)
    is_done = False
    state = env.reset()
    while not is_done:        # Loop until game ends
        env.render()
        # subprocess.call('clear', shell=False)
        video_recorder.capture_frame()                                  
        action = policy[state]
        state, reward, is_done, info = env.step(action)
    video_recorder.close()
    video_recorder.enabled = False
    env.close()
    rec.make_gif_from_gym_record(video_dir)

if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    rec = string_recorder.StringRecorder()  
    timestep_limit = 30
    video_dir = "./video/taxi2.json"
    video_recorder = VideoRecorder(env, video_dir, enabled=True)
    is_done = False
    state = env.reset()
    while not is_done:        # Loop until game ends
        env.render()
        # subprocess.call('clear', shell=False)
        video_recorder.capture_frame()                                  
        action = env.action_space.sample()
        state, reward, is_done, info = env.step(action)
    video_recorder.close()
    video_recorder.enabled = False
    env.close()
    rec.make_gif_from_gym_record(video_dir)