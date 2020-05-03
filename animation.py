import numpy as np

from config import *

import matplotlib.pyplot as plt
from matplotlib import animation

# Animates a model given a number of frames and number of steps per frame
# Steps per frame should be on the order of L^2, where L is the size of the model
def animate_model(model, nframes, nsteps, save=False, title="animation.mp4"):
    fig, ax = plt.subplots(figsize=(6,6))

    steps = 1000

    im = ax.imshow(model.s)

    def init():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("T = " + str(model.T), fontsize=15)
        return (im,)

    def update(frame):
        model.step(nsteps)
        im.set_array(model.s)
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, nframes),
                                  init_func=init, blit=True)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15)

        ani.save(title, writer=writer)
    else:
        plt.show()


L = 100

model = Ising2DTF(L, B=0., I=1., T=1.)
animate_model(model, 300, 3000, save=True, title="ferro.mp4")

model = Ising2DTF(L, B=0., I=1., T=100.)
animate_model(model, 300, 3000, save=True, title="para.mp4")


