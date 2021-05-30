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
   - SARSA had similar problems, but to a lesser extent. and grid search resulted in an average increase in reward of +12.
   - **Double Q-Learning** worked far better than Q-Learning as it did not overestimate action values as much. However, this method tended to somewhat underestimate action values, and so converged more slowly. Nonetheless, episode rewards converged better than Q-Learning (but SARSA + Grid Search still outperformed). These methods depend greatly on hyperparameter selection.
     
     
