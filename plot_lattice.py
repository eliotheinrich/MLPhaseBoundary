from config import *
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_checkerboard(ax,s):
    L = s.shape[0]
    ax.set_xlim(-.05*L, L)
    ax.set_ylim(-.05*L, L)
    scale = 300
    for i in range(0,L):
        for j in range(i%2,L,2):
            if s[i][j] == 1: facecolor='k'
            else: facecolor='none'

            ax.scatter(i,j, s=scale, marker='o', facecolors=facecolor, edgecolors='k')

    return ax


if __name__=="__main__":
    state_file = open(r'data/Ising_state_sq_ice.pkl', 'rb')
    states = pickle.load(state_file)
    state_file.close()

    fig,ax = plt.subplots(figsize=(6,6))
    ax.set_xticks([])
    ax.set_yticks([])

    num = int(input("State number: "))

    plot_checkerboard(ax, states[num])
    plt.title("State " + str(num), fontsize=16)
    plt.show()




