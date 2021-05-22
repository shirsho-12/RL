# Reinforcement Learning

Small projects to delve into reinforcement learning. Following Sutton & Barto's Reinforcement Learning: An Introduction.

1. Install dependencies from requirements.txt
2. Run the random search algorithm on the CartPole environment. Works well due to the simplicity of the environment: Only 2 possible actions and 4 state parameters to consider.
3. Perform hill-climb (gradient descent) and policy gradient on the CartPole environment. Both work well.
4. Run value iteration and policy iteration on the FrozenLake environment. Both methods are suitable for the 4x4 and 8x8 
   cases. Random search does not work at all, with an abysmal sub 1% success rate.
5. Run value iteration on the biased coin gambling problem. Policy iteration fails to converge as can be seen 
   in my notebook environment. There appears to be a bug in the policy evaluation function which I cannot pinpoint as of now.
   