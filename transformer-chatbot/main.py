import os
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

ROOT = os.path.dirname(__file__)

api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    'projjal1/human-conversation-training-data',
    path=os.path.join(ROOT, 'data'),
    unzip=True
)

with open(os.path.join(ROOT, 'data', 'human_chat.txt'), 'r') as data_file:
    data = data_file.readlines()
    human1, human2 = [], []

    for line in data:
        if line.startswith('Human 1'):
            human1.append(line.replace('Human 1:', '').replace('\n', ''))
        else:
            human2.append(line.replace('Human 2:', '').replace('\n', ''))

    total_lines = min(len(human1), len(human2))
    
train_examples = pd.DataFrame()
train_examples['human_1'] = human1[:total_lines]
train_examples['human_2'] = human2[:total_lines]

print(train_examples)

train_human1 = tf.data.Dataset.from_tensor_slices(train_examples['human_1'].values)
train_human2 = tf.data.Dataset.from_tensor_slices(train_examples['human_2'].values)

bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ['[PAD]', '[UNK]', '[START]', '[END]']

bert_vocab_args = dict(
    vocab_size=8000,
    reserved_tokens=reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={}
)

def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)

human1_vocab = bert_vocab.bert_vocab_from_dataset(
    train_human1.batch(1000).prefetch(2),
    **bert_vocab_args
)

human1_vocab_path = os.path.join(ROOT, 'data', 'human1_vocab.txt')
write_vocab_file(human1_vocab_path, human1_vocab)

human2_vocab = bert_vocab.bert_vocab_from_dataset(
    train_human2.batch(1000).prefetch(2),
    **bert_vocab_args
)

human2_vocab_path = os.path.join(ROOT, 'data', 'human2_vocab.txt')
write_vocab_file(human2_vocab_path, human2_vocab)

human1_tokenizer = text.BertTokenizer(human1_vocab_path, **bert_tokenizer_params)
human2_tokenizer = text.BertTokenizer(human2_vocab_path, **bert_tokenizer_params)

# demonstrate tokenizer
for p in train_human1.batch(3).take(1):
    for ex in p:
        print(ex.numpy())

    token_batch = human1_tokenizer.tokenize(p)
    token_batch = token_batch.merge_dims(-2, -1)

    for ex in token_batch.to_list():
        print(ex)

    txt_tokens = tf.gather(human1_vocab, token_batch)
    print(tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1))
    words = human1_tokenizer.detokenize(token_batch)
    print(tf.strings.reduce_join(words, separator=' ', axis=-1))
       