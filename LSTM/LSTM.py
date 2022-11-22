import torch
import torchtext
import torch.nn as nn
from torch.autograd import Variable
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.optim as optim
import warnings
import os
import sys
warnings.filterwarnings('ignore')

TEXT = data.Field(include_lengths=True)

# If you want to use English tokenizer from SpaCy, you need to install SpaCy and download its English model:
# pip install spacy
# python -m spacy download en_core_web_sm
# TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)

LABEL = data.LabelField(dtype=torch.long)
train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL, train_subtrees=True, filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train_data)
# Here, you can also use some pre-trained embedding
# TEXT.build_vocab(train_data,
#                  vectors="glove.6B.100d",
#                  unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: len(x.text),
    batch_size=batch_size, device=device)

embedding_size = 128
hidden_size = 128
vocab_size = len(TEXT.vocab.itos)
num_classes = 1
num_epoch= 20

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.word_vec = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(embedding_size, hidden_size, 1, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
 
    def forward(self, input):
        embedding_input = self.word_vec(input)
        embedding_input = embedding_input.permute(1, 0, 2)
        output, (h_n, c_n) = self.bilstm(embedding_input)
        encoding1 = torch.cat([h_n[0], h_n[1]], dim=1)
        encoding2 = torch.cat([output[0], output[-1]], dim=1)
        fc_out = self.fc(encoding1).squeeze()
        return fc_out
 
model = BiLSTM()

class LogisticLoss(nn.Module):
    def __init__(self):
        super(LogisticLoss, self).__init__()
    def forward(self, inputs, target):
        return torch.mean(torch.log(1.0/torch.sigmoid(target * inputs)))

criterion = LogisticLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, momentum=0.9)

fname = sys.path[0] + os.sep + "log" + os.sep + "LSTM_loss.txt"

with open(fname, 'w') as f:
    for epoch in range(num_epoch):
        total_loss = 0
        for i, ((encode, length), labels) in enumerate(train_iterator):
            labels = Variable(2 * (labels.float() - 0.5))
            pred = model(encode.transpose(0,1))
            loss = criterion(pred, labels)
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epoch, i + 1,
                        len(train_iterator), loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Print your results every epoch
        epoch_loss = total_loss / len(train_iterator)
        correct = 0
        total = 0
        for i, ((encode, length), labels) in enumerate(test_iterator):
            labels = Variable(2 * (labels.float() - 0.5))
            outputs = model(encode.transpose(0,1))
            predicted = torch.where(torch.sigmoid(outputs)>0.5, 1, -1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        f.write('Epoch: [% d/% d]: Loss: %.4f , Accuracy of the currenent model: % .4f %%\n' % (
            epoch + 1, num_epoch,
            epoch_loss,
            100.0 * correct / total))