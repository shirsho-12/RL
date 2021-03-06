# Reinforcement Learning

Small projects to delve into reinforcement learning. Following Sutton & Barto's Reinforcement Learning: An Introduction.

## Developments 

### Basics (17 May 2021)
1. Install dependencies from requirements.txt
2. Run the **random search** algorithm on the CartPole environment. Works well due to the simplicity of the environment: Only 2 possible actions and 4 state parameters to consider.
3. Perform **hill-climb** (gradient descent) and **policy gradients** on the CartPole environment. Both work well.

### Dynamic Programming (22 May 2021)
4. Run **policy iteration** and **value iteration** on the FrozenLake environment. Both methods are suitable for the <img src="https://render.githubusercontent.com/render/math?math=4\times 4"> and <img src="https://render.githubusercontent.com/render/math?math=8\times 8"> cases. Random search does not work at all, with an abysmal sub-1% success rate.
5. Run value iteration on the biased coin gambling problem. Policy iteration fails to converge as can be seen in my notebook. There appears to be a bug in the policy evaluation function which I cannot pinpoint as of now.
   
### Monte Carlo (25 May 2021)
6. Estimate the value of pi using the dots in a square method. Very basic Monte Carlo.
7. Estimate the reward value of the DP optimal policy using **first-visit** and **every-visit policy evaluation**. Estimated values are very close to DP exact values.
8. Find the optimal policy for blackjack, a game with <img src="https://render.githubusercontent.com/render/math?math=280\times 280 \times 2"> states with on-policy and off-policy Monte Carlo control.
   - For on-policy Monte Carlo control, exploring starts and <img src="https://render.githubusercontent.com/render/math?math=\epsilon">**-greedy** soft policies yield good results, with <img src="https://render.githubusercontent.com/render/math?math=\epsilon">-greedy outperforming exploring 
     starts over more iterations.
   - In off-policy Monte Carlo control, **incremental Q-function** updates greatly increases scalability and reduces space 
     complexity over the naive implementation. In addition, **weighted importance sampling** improves accuracy and yields consistent
     results compared to the high variance seen in ordinary frequency sampling.

### Temporal Difference Learning (30 May 2021)
9. Find the optimal policy of the Cliff Walking environment using **Q-learning**. A simple problem that Q-learning solved in relatively few episodes.
10. Find the optimal policy of Windy GridWorld, a similar environment to Cliff Walking but with additional wind displacement. **SARSA** worked great here, with Q-learning showing similar results. 
11. For the taxicab problem 3 methods were tried:
   - Q-Learning converged with a relatively high error rate, as could be seen in the graphs. Grid search will probably reduce error even further. My attempts did not result in significant improvement even after stepwise <img src="https://render.githubusercontent.com/render/math?math=\alpha"> reductions after 400 episodes and increasing episode lengths.
   - SARSA had similar problems, but to a lesser extent. Grid search resulted in an average increase in reward of +12.
   - **Double Q-Learning** worked far better than Q-Learning as it did not overestimate action values as much. However, this method tended to somewhat underestimate action values, and so converged more slowly. Nonetheless, episode rewards converged better than Q-Learning (but SARSA + Grid Search still outperformed). These methods depend greatly on hyperparameter selection.
     
### Multi-Arm Bandits (6 July 2021)
12. Solve multi-arm bandit (slot machines with probabilistic rewards) problems using random, <img src="https://render.githubusercontent.com/render/math?math=\epsilon">-greedy, and softmax policies. Here, the <img src="https://render.githubusercontent.com/render/math?math=\epsilon">-greedy policy converged the fastest.
13. Solve a simplified webpage ad placement problem using **Upper Confidence Interval (UCB)** and **Thompson Sampling**. In this case, *contextual bandits* (one bandit leading to another) were used. This increased the information and complexity of the problem.

### Function Approximation (10 July 2021)
14. Estimate Q-values using linear equations, reducing the dimensionality of the problem. These linear equations are first solved using a fully connected layer `nn.Linear`. This is then improved into a shallow neural network with `nn.Sequential` and ReLU non-linearities `nn.ReLU()`. 
15. Approximate Q-values are then used to solve continuous state environments like the CartPole and MountainCar environments. An improvement on Q-learning is done using experience replay. Experience replay improves learning by randomly choosing a number of states to learn from instead of the entire episode. 