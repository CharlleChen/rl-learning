import numpy as np
# from render import MainWindow
# from PySide6.QtWidgets import QApplication
# from threading import Thread
from IPython import embed as e
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, rows, cols, render=True):
        self.rows = rows
        self.cols = cols

        # initialize states
        self.states = np.array([[i, j] for i in range(rows) for j in range(cols)])
        self.obs_dim = 2 # or row * cols
        self.actions = [0, 1, 2, 3]
        self.act_dim = len(self.actions)


        self._start = [0, 0]

        self.end = [
            [np.array([0, cols-1]), -1],
            [np.array([rows-2, cols-1]), 1]
        ]

        self.done = False

        # self._state = self._start

        # self.fig, self.ax = plt.subplots(figsize=(5, 5))
        # self.p = self.ax.imshow(self.v)

        # self.renderer = Thread(target=app.exec_)


    @property
    def start(self):
        return self._start.copy()


    """
    0: down
    1: up
    2: left
    3: right
    Stay in the position if bound on the wall
    """
    def _move(self, state, action):
        new_state = state.copy()

        if action == 0:
            new_state[0] += 1
        elif action == 1:
            new_state[0] -= 1
        elif action == 2:
            new_state[1] -= 1
        elif action == 3:
            new_state[1] += 1
        else:
            e()
            b
            print("Unexpected moving. Exiting!")
            exit(0)
        
        new_state[0] = np.clip(new_state[0], 0, self.rows - 1)
        new_state[1] = np.clip(new_state[1], 0, self.cols - 1)

        return new_state
    
    # Deterministic move
    def P(self, old_state, new_state, action):
        # e()
        if (self._move(old_state, action) == new_state).all():
            return 1
        else:
            return 0
        

    def R(self, last_state, new_state, action):
        for i in self.end:
            if (last_state == i[0]).all():
                return i[1]

        return 0
    
    def is_done(self, state):
        for i in self.end:
            if (state == i[0]).all():
                return True
        return False
    
    def get_new_state(self, state):
        new_states = []
        for action in self.actions:
            new_states.append(self._move(state, action))
        return new_states

    def step(self, action):
        if self.done:
            print("Already done. Do nothing. Reset to continue")
            return [None, None, None]
        new_state = self._move(self._state, action)

        r = self.R(self._state, new_state, action)
        done = self.is_done(self._state)
        self.done = done
        self._state = new_state

        return [self._state.copy(), r, done]
    
    def reset(self):
        self._state = self.start
        self.done = False

        return self._state.copy()

    def print(self):
        print("-" * self.cols * 4)
        for r in range(self.rows):
            for c in range(self.cols):
                mark = "o"
                for end in self.end:
                    if [r,c] == end[0].tolist():
                        mark = str(end[1])
                        break
                if self._state==[r,c]:
                    mark = 'x'
                print(f"| {mark} " , end='')
            print("|")
        print("-" * self.cols * 4)
    
    # def plot(self):
    #     self.v[0,1] += 10
    #     self.p.set_data(self.v)
    #     plt.pause(0.5)
    
    def get_V(self):
        V = np.zeros((self.rows, self.cols))
        for i in self.end:
            V[i[0][0], i[0][1]] = i[1]
        return V



if __name__ == '__main__':
    env = Environment(5, 5)
    # plt.pause(0.5)
    # env.renderer.start()
    
    state = env.reset()

    while True:
        env.print()
        action = int(input('Input action:\n'))

        if action == 4:
            env.reset()
            state = env.start
            print("RESET", state)
            continue

        [new_state, r, done] = env.step(action)

        state = new_state if new_state else state
        print(state, r, done)

    # renderer.start()
