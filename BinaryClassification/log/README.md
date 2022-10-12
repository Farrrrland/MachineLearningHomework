# Log Folder
This folder contains all the output log file for the training process, including:

1. Average losses of each epoch
2. Accuracy of each Epoch

Each model have a bunch of log files based on the different optimizer type it uses, which is indicated with different suffix of the file (e.g., _0_1.txt means this is the log file of using SGD and with the second learning rate defined in the code, _1_2.txt means using SGD-Momentum and with the third learning rate.)