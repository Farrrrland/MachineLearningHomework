# LSTM Sentence Emotion Analysis
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Farrrrland/MachineLearningHomework/HEAD)
## Content and file structure
According to the reguirements (see `./file` folder), this program includes:

1. Report and requirement documents of this program can be found in `./files` folder.
2. `LSTM.py`, source code of the analyser. We imply a sequence model with LSTM to conduct sentiment analysis on SST-2 dataset with postive and negative sentiments. The loss and accuracy of the model of each epoch is reported the same way as the project implemented before.
3. `plot.py` is used to plot the results of the analyser, including losses of each epoch.
4. `./img` folder contains the output of `plot.py`.
5. `./log` folder stores the log data for the training process, including the average loss and accuracy of each iteration. In this approach, epoch size is set to 20, and the last piece of the log data contains the final accuracy of the model.


## How to start
Note that due to the severe version conflict of the packages from the given text preprocessing demo code using `torchtext`, this project is inplemented in through repo2dcker with [Binder](https://mybinder.org/).

By modifying  `../requirements.txt` (see [requirements](main/requirements.txt)), we condition the version of `torch==1.8.0` and `torchtext==0.9.0` according to the demo code and [version requirements](https://pypi.org/project/torchtext/)

### Launch
First, launch binder as shown at top of this file and wait for repo2docker to push the docker image with the required packages.

Then you will access the container generated from this repo with Jupyter Lab, move to `./LSTM` folder and create a terminal.

After that, simply run the code with 
```bash
python3 LSTM.py
python3 plot.py
```

You can find the figure in `./img` folder and log data in `./log` folder afterwards.