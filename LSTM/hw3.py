import torch
import torchtext
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import torch.optim as optim
import warnings
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