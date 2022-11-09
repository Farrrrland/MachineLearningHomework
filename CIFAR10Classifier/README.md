# CIFAR-10 Image Classifier
## Content and file structure
According to the reguirements (see `/file` folder), this program includes:

1. Report and requirement documents of this program can be found in `./files` folder.
2. `image_classifier.py`, source code of the cifar-10 image classification pipeline. We imply function `runModel` and by configuration, supports different types of neural networks to run the classification process, which includes CNN, MLP and MLP with some modifications. We want to compare the different results of different network sgtructures. 
3. `ploy.py` is used to plot the results of the classifier, including losses of each epoch under different network structures.
4. `./img` folder contains the out put of `plot.py`.
5. `./log` folder stores the log data for the training process, including the average loss and accuracy of each iteration. The log files are organized by models, each model has one unique log file which is easy to identify. In this approach, epoch size is set to 20, and the last piece of the log data contains the final accuracy of the model.


## How to start

Simply run the code with 
```bash
# run image classfier with different model, and record the data needed
python3 image_classifier.py
# plot the loss data of each epoch
python3 plot.py
```

Then you can find the figure in `./img` folder and log data in `./log` folder.