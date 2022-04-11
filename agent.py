import numpy as np
from environment import Environment
from IPython import embed as e
import matplotlib.pyplot as plt

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
        return self.probs[state[0], state[1], action]
    
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
    # print(plt.get_cmap('plasma'))
    p = ax.imshow(V, cmap=plt.get_cmap('plasma'))
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

        ax.text(j_, i_, disp_string + 'v={:0.2f}\n{}'.format(d, action_map[policy.act([i_, j_])]), ha='center', va='center',
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
        

if __name__ == '__main__':
    policy = RandomWalk()
    n=10
    env = Environment(n, n)

    agent = DetermPolicy((n,n))
    # print(agent.probs)

    
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

    