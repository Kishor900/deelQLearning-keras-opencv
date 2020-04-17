from game.env import Player
import numpy as np
from collections import deque
import tensorflow as tf
import random
import time

memory = deque(maxlen=2000)
batch_size = 32
p = Player()
state = np.array((0, 0, 0, 0)).reshape((1, 4))
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95
t1 = time.time()
load_model = True

if not load_model:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(24, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(4, activation='linear')])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
else:
    model = tf.keras.models.load_model("model.h5")
    memory = deque(list(np.load("memory.npy")), maxlen=2000)
    print(len(memory))


def act(state):
    if np.random.rand() <= epsilon:
        return np.random.choice([0, 1, 2, 3])
    act_values = model.predict(state)
    return np.argmax(act_values[0])  # returns action


def replay():
    global epsilon
    global epsilon_min
    global epsilon_decay
    global t1
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state in minibatch:
        target = (reward + gamma * np.amax(model.predict(next_state)[0]))  # belman equation
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    t2 = time.time()
    if t2 - t1 >= 120:
        t1 = t2
        print("saving a model model.h5")
        model.save("model.h5")
        print("saving memory memory.npy")
        np.save("memory", memory)


episode = 0

while True:
    p.spawn()
    while True:
        action = act(state)
        p.action = action
        p.run()
        next_state, reward = p.nextState, p.reward
        next_state = np.array(next_state).reshape((1, 4))
        state = np.array(state).reshape((1, 4))
        state = next_state
        memory.append((state, action, reward, next_state))
        if len(memory) >= batch_size:
            replay()
        else:
            print("filling batch up to size 32")

        if p.done:
            print("episode:", episode, epsilon)
            break
    episode = episode + 1
