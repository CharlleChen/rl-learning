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
        self.actions = [0, 1, 2, 3]

        self.v = np.zeros((rows, cols))

        self._start = [0, 0]

        self.end = [
            [np.array([rows-1, cols-1]), 1],
            [np.array([rows-2, cols-1]), -1]
        ]

        self.done = False

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
    """
    def move(self, state, action):
        last_state = state.copy()
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
            print("Unexpected moving. Exiting!")
            exit(0)
        
        new_state[0] = np.clip(new_state[0], 0, self.rows - 1)
        new_state[1] = np.clip(new_state[1], 0, self.cols - 1)

        return [last_state, new_state]
    
    # Deterministic
    def P(self, old_state, new_state, action):
        # e()
        if (self.move(old_state, action)[1] == new_state).all():
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
            new_states.append(self.move(state, action)[1])
        return new_states

    def step(self, state, action):
        if self.done:
            print("Already done. Do nothing. Reset to continue")
            return [None, None, None]
        [last_state, new_state] = self.move(state, action)

        r, done = self.R(last_state, new_state, action)
        self.done = done

        return [new_state, r, done]
    
    def reset(self):
        # self.state = self.start
        self.done = False

    def print(self):
        print("-" * self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                print(self.v[r, c], end='\t')
            print()
        print("-" * self.cols)
    
    def plot(self):
        self.v[0,1] += 10
        self.p.set_data(self.v)
        plt.pause(0.5)
    
    def get_V(self):
        V = np.zeros((self.rows, self.cols))
        for i in self.end:
            V[i[0][0], i[0][1]] = i[1]
        return V



if __name__ == '__main__':
    env = Environment(5, 5)
    # plt.pause(0.5)
    # env.renderer.start()
    
    state = env.start

    while True:
        env.print()
        action = int(input('Input action:\n'))

        if action == 4:
            env.reset()
            state = env.start
            print("RESET", state)
            continue

        [new_state, r, done] = env.step(state, action)

        state = new_state if new_state else state
        print(state, r, done)

    # renderer.start()
