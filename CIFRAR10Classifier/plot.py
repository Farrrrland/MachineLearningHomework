import matplotlib.pyplot as plt
import sys
import os


plt.rcParams['font.size'] = '25'

def pltLoss(fname):
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=500)
    fname = f"{fname}.txt"
    losses = []
    for line in open(sys.path[0] + os.sep + "log" + os.sep + fname):
        losses.append(float(line.split()[5]))
    ax.plot(range(len(losses)), losses, lw=2)
    ax.set_xlabel("Number of Epoch", size=28)
    ax.set_ylabel("Average Loss", size=28)
    plt.title(fname.split('.')[0].replace('_', ' ').upper(), size=32)
    plt.tight_layout()
    # plt.legend(fontsize=20)
    plt.savefig(sys.path[0] + os.sep + "img" + os.sep +f"losses_{fname.split('.')[0]}.pdf")

if __name__ == "__main__":
    pltLoss("mlp_non_act")
    pltLoss("mlp")
    pltLoss("cnn")