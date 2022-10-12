# Binary Classification

## Content and file structure
According to the requirements, this program includes

1. Report and other documents of this program can be found in `./files` folder.
2. `binary_classification.py`, source code of the image classification pipeline. We imply function `runModel` and by configuration, supports two types of classifier, logistic regression or the SVM. Multiple step sizes are included for detailed analysis.
3. `ploy.py` is used to plot the results of the classifier, including losses of each epoch (iteration) with different hyper parameters.
4. `./img` folder contains the out put of `plot.py`.
5. `./log` folder stores the log data for the training process, including the average loss and accuracy of each iteration. The log files are organized by model type and hyper parameters, see `/log/README.md` for more details. In this approach, epoch size is set to 50, and the last piece of the log data contains the final accuracy of the model.

## How to start

Simply run the code with 
```bash
# run binary classification with logistic regression and svm, and record the data needed
python3 binary_classificartion.py
# plot the loss data of each epoch
python3 plot.py
```

Then you can find the figure in `./img` folder and log data in `./log` folder.