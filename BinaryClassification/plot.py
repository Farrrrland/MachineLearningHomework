import matplotlib.pyplot as plt
import sys
import os


momentums = [0, 0.9]
learning_rates = [0.001, 0.01, 0.1]
lines = ['-', '-.']

def pltLoss(model_type):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    for idx, momentum in enumerate(momentums):
        for idx_, learning_rate in enumerate(learning_rates):
            fname = f"epoch_{model_type}_{idx}_{idx_}.txt"
            losses = []
            for line in open(sys.path[0] + os.sep + "log" + os.sep + fname):
                losses.append(float(line.split()[5]))
            label = "SGD"
            if idx != 0:
                label += f"_Momentum_{momentum}"
            label += f"_lr_{learning_rate}"
            ax.plot(range(len(losses)), losses, label = label, ls = lines[idx])
    ax.set_xlabel("Number of Epoch")
    ax.set_ylabel("Average Loss")
    plt.title(model_type.replace("_"," ").title())
    plt.tight_layout()
    plt.legend()
    plt.savefig(sys.path[0] + os.sep + "img" + os.sep +f"losses_{model_type}.pdf")

def pltAccuracy(model_type):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    for idx, momentum in enumerate(momentums):
        for idx_, learning_rate in enumerate(learning_rates):
            fname = f"epoch_{model_type}_{idx}_{idx_}.txt"
            accuracy = []
            for line in open(sys.path[0] + os.sep + "log" + os.sep + fname):
                accuracy.append(float(line.split()[12]))
            label = "SGD"
            if idx != 0:
                label += f"_Momentum_{momentum}"
            label += f"_lr_{learning_rate}"
            ax.plot(range(len(accuracy)), accuracy, label = label, ls = lines[idx])
    ax.set_xlabel("Number of Epoch")
    ax.set_ylabel("Accuracy (%)")
    plt.title(model_type.replace("_"," ").title())
    plt.tight_layout()
    plt.legend()
    plt.savefig(sys.path[0] + os.sep + "img" + os.sep +f"accuracy_{model_type}.pdf")


if __name__ == "__main__":
    pltLoss("logistic_regression")
    pltLoss("svm")
    pltAccuracy("logistic_regression")
    pltAccuracy("svm")