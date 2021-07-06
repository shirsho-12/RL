import torch


# Random policy: naive, not the most refined (or accurate)
def get_random_policy(num_actions):
    def policy():
        action = torch.multinomial(torch.ones(num_actions), 1).item()
        return action
    return policy


# epsilon-greedy policy
def gen_eps_greedy_policy(num_actions, epsilon):
    def policy(Q):
        probs = torch.ones(num_actions) * epsilon / num_actions
        best_action = torch.argmax(Q).item()
        probs[best_action] = 1. - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy


# Softmax policy
def gen_softmax_exploration_policy(tau):
    def policy(Q):
        probs = torch.exp(Q / tau)
        probs /= torch.sum(probs)
        action = torch.multinomial(probs, 1).item()
        return action
    return policy


# Upper confidence bound algorithm
def upper_confidence_bound(Q, action_count, t):
    # t -> episode number 
    ucb = torch.sqrt((2 * torch.log(torch.tensor(float(t)))) / action_count) + Q
    return torch.argmax(ucb).item() 


# Thompson Sampling
def thompson_sampling(alpha, beta):
    prior_values = torch.distributions.beta.Beta(alpha, beta).sample()
    return torch.argmax(prior_values)
