import matplotlib.pyplot as plt
import sys
import os


momentums = [0, 0.8, 0.9, 0.95, 0.98]

# Losses of each epoch with different optimizer using Logistic Regression
fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
for idx, momentum in enumerate(momentums):
    fname = f"epoch_logistic_regression_{idx}.txt"
    losses = []
    for line in open(sys.path[0] + os.sep + "log" + os.sep + fname):
        losses.append(float(line.split()[5]))
    label = "SGD"
    if idx != 0:
        label += f"_Momentum_{momentums[idx]}"
    ax.plot(range(len(losses)), losses, label = label)
ax.set_xlabel("Number of Epoch")
ax.set_ylabel("Average Loss")
plt.title("Logistic Regression")
plt.tight_layout()
plt.legend()
plt.savefig(sys.path[0] + os.sep + "img" + os.sep + "losses_logistic_regression.pdf")

ax.clear()

# Using SVM
fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
for idx, momentum in enumerate(momentums):
    fname = f"epoch_svm_{idx}.txt"
    losses = []
    for line in open(sys.path[0] + os.sep + "log" + os.sep + fname):
        losses.append(float(line.split()[5]))
    label = "SGD"
    if idx != 0:
        label += f"_Momentum_{momentums[idx]}"
    ax.plot(range(len(losses)), losses, label = label)
ax.set_xlabel("Number of Epoch")
ax.set_ylabel("Average Loss")
plt.title("Support Vector Machine")
plt.tight_layout()
plt.legend()
plt.savefig(sys.path[0] + os.sep + "img" + os.sep + "losses_svm.pdf")





# ax.xaxis.set_major_locator(plt.MultipleLocator(5))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# ax.set_xlim(0, 20)