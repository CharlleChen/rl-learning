import numpy as np
from environment import Environment
from IPython import embed as e
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch
import argparse

action_map = ['down', 'up', 'left', 'right']

np.set_printoptions(precision=3)

class RandomWalk:
    def __init__(self):
        self.options = [0, 1, 2, 3]

    def prob(self, state, action):
        return 1/len(self.options)
    
    def backprop(self, loss):
        pass

class DetermPolicy:
    def __init__(self, shape_of_states):
        self.options = [0, 1, 2, 3]
        self.probs = np.zeros((*shape_of_states, len(self.options)))
        for r in range(self.probs.shape[0]):
            for c in range(self.probs.shape[1]):
                act = np.random.choice([1,3])
                self.probs[r][c][act] = 1

    def prob(self, state, action):
        return self.probs[int(state[0]), int(state[1]), action]
    
    def act(self, state):
        action = -1
        act_prob = 0
        for a in self.options:
            p = self.prob(state, a)
            if p > act_prob:
                action = a
                act_prob = p
        return action
    
    # deterministic update
    def update(self, state, action):
        for a in self.options:
            if a == action:
                self.probs[state[0], state[1], a]=1
            else:
                self.probs[state[0], state[1], a]=0
    
    def backprop(self, loss):
        pass

"""
    Dynamic Programming
"""

def policy_evaluation(env, policy, gamma=0.9, tol = 1e-4):
    env.reset()

    V = env.get_V()
    # print("V:\n", V)
    
    for i in range(1000):
        # print(f"iter {i}")
        # print("V:\n", V)
        delta = 0
        for s in env.states:
            # Deal the finish case as a corner case
            if env.is_done(s):
                V[s[0], s[1]] = env.R(s, s, None)
                continue

            last_V = V[s[0], s[1]]
            new_V = 0
            for a in env.actions:

                for new_state in env.get_new_state(s):
                    new_V += policy.prob(s, a) * env.P(s, new_state, a) * (env.R(s, new_state, a) + gamma * V[new_state[0], new_state[1]])

            
            V[s[0], s[1]] = new_V
            delta = max(delta, np.abs(last_V - V[s[0], s[1]]))
        if abs(delta) < tol:
            break
            # V[s[0], s[1]] = 

    return V

def plot_grid(V, env, policy, fig_size=8):
    # Visualize
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    # # print(plt.get_cmap('plasma'))
    print(V)
    p = ax.imshow(V)
    # fig = plt.gcf()
    # ax = plt.gca()
    for (i_, j_), d in np.ndenumerate(V):
        s = np.array([i_, j_])
        disp_string = ""
        if (s == env.start).all():
            disp_string += "Start\n"
        else:
            for end_state in env.end:
                if (s == end_state[0]).all():
                    disp_string += "End\n"

        ax.text(j_, i_, disp_string + 'v={:0.2f}\n{}'.format(d, action_map[policy.act(torch.tensor([i_, j_], dtype=torch.float32))]), ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax.title.set_text("Simple grid world")
    plt.show()
    

def policy_improvement(env, V, policy, gamma=0.9):
    for i in range(1000):
        # print(f"iter {i}")

        policy_stable = True
        for s in env.states:
            if env.is_done(s):
                continue
            
            old_act = policy.act(s)
            new_act = -1
            new_v = -1
            for a in env.actions:
                sum_v = 0

                for new_state in env.get_new_state(s): # iterate through all possible new_states
                    sum_v += env.P(s, new_state, a) * (env.R(s, new_state, a) + gamma * V[new_state[0], new_state[1]])
                # if the path is better
                if sum_v > new_v:
                    new_act = a
                    new_v = sum_v
            
            # ending condition
            if new_act != old_act:
                policy_stable = False
                policy.update(s, new_act)
        if policy_stable: break

    return policy

"""
    Vanilla Gradient Policy
"""

def mlp(sizes, activation, output_act=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_act
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, activation=nn.Tanh):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) +[action_dim], activation)

    def _distribution(self, obs):
        logits = self.net(obs)
        return Categorical(logits = logits)
    
    # def _log_prob(self, obs, act):
    #     pi = self._distribution(obs)
    #     return pi.log_prob(act)

    def forward(self, obs, action=None):
        pi = self._distribution(obs)

        logp_a = None
        if action is not None:
            logp_a = pi.log_prob(action)
        return pi, logp_a

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self,obs):
        return torch.squeeze(self.net(obs), -1)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, act_dim):
        super().__init__()

        activation = nn.Tanh

        self.pi = Actor(obs_dim, hidden_sizes, act_dim, activation)
        self.v = Critic(obs_dim, hidden_sizes, activation)
    
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()

            log_prob_a = pi.log_prob(a)
            v = self.v(obs)

        return a.numpy(), v.numpy(), log_prob_a.numpy()
    
    def act(self, obs):
        return self.step(obs)[0]


def vgp(env:Environment, hidden_sizes=[64, 64], epochs=50, batch_size=32, lr=1e-2):
    model = ActorCritic(env.obs_dim, list(hidden_sizes), env.act_dim)

    actor_optim = torch.optim.AdamW(model.pi.parameters(), lr = lr)
    critic_optim = torch.optim.AdamW(model.v.parameters(), lr = lr)

    
    def compute_pi_loss(model, obs, action, weights):
        obs = torch.tensor(obs,dtype=torch.float32)
        action = torch.tensor(action,dtype=torch.float32)
        weights = torch.tensor(weights,dtype=torch.float32)
        logp = model.pi(obs, action)[1]
        return -(logp * (weights + 1)).mean()


    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rew = []
        batch_lengths = []
        batch_vals = []

        state = env.reset()

        eps_rew = []
        eps_vals = []

        while True:
            act, v, log_prob_a = model.step(torch.tensor(state, dtype=torch.float32))
            new_state, rewards, done = env.step(act)
            
            eps_rew.append(rewards)
            eps_vals.append(v)

            batch_acts.append(act)
            batch_obs.append(state.copy())

            state = new_state

            if done or len(eps_rew) > 100:
                state = env.reset()

                eps_len = len(eps_rew)
                eps_sum = np.sum(eps_rew)

                # finish_path(eps_rew, eps_vals)

                # Advantage function
                # eps_rew -= eps_vals
       
                # What is weights?
                batch_weights += [eps_sum] * eps_len
               
                # batch_weights += eps_rew
        
                batch_lengths.append(eps_len)
                batch_rew.append(np.sum(eps_rew))

                eps_rew = []


                if len(batch_obs) > batch_size:
                    break
        
        # Compute loss and update parameters after certain batche size
        actor_optim.zero_grad()
        pi_batch_loss = compute_pi_loss(
            model = model,
            obs = np.array(batch_obs),
            action = np.array(batch_acts),
            weights = np.array(batch_weights)
        )
        pi_batch_loss.backward()
        actor_optim.step()



        return pi_batch_loss, batch_rew, batch_lengths

    for epoch in range(epochs):
        batch_loss, batch_rew, batch_lengths = train_one_epoch()
        print("epoch %3d \t loss %.2f \t return %.2f \t lengths %.2f " \
                % (epoch, batch_loss, np.mean(batch_rew), np.mean(batch_lengths)))
    plot_grid(env.get_V(), env, model)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='pi', help="[pi, vgp]")
    args = parser.parse_args()

    policy = RandomWalk()
    n=10
    env = Environment(n, n)

    agent = DetermPolicy((n,n))
    # print(agent.probs)

    if args.policy=='vgp':
        vgp(env, [256, 256], epochs=100, lr=1e-5)
    elif args.policy=='pi':
    # exit(0)
    
        V = policy_evaluation(env, agent)
        for i in range(20):
            print(f"Iteration {i}")
            last_V = V.copy()
            agent = policy_improvement(env, V, agent)
            V = policy_evaluation(env, agent)

            print("Change:", np.linalg.norm(V - last_V))
            if np.linalg.norm(V - last_V) < 1e-4:
                break
        plot_grid(V, env, agent)
    else:
        print("Implemented algorithms are pi (policy iteration) and vgp (Vanilla Gradient Policy)")

    