#------------------------------------------------------------------------------------------
# 모듈 로딩
#------------------------------------------------------------------------------------------

# Pytorch 관련 모듈
from torch import nn
from torch import optim
import torch
from torch.utils.data import TensorDataset, DataLoader

# 자연어 처리 관련 모듈 
from konlpy.tag import Okt

# 모델링 관련 모듈
from gensim.models import Word2Vec

# 데이터셋 관련 모듈
from Korpora import Korpora

# 데이터분석 관련 모듈
import pandas as pd
import numpy as np

# 파이썬 내장 모듈
from collections import Counter

# 사용자 정의 모듈
from funcRNN import *
from classRNN import *

#------------------------------------------------------------------------------------------
# 1. 데이터준비
#------------------------------------------------------------------------------------------

corpus = Korpora.load('nsmc')
corpus_df = pd.DataFrame(corpus.test)

train = corpus_df.sample(frac=0.9, random_state=42)
test = corpus_df.drop(train.index)
print(train.head(5).to_markdown())
print("Training Data Size :", len(train))
print("Testing Data Size :", len(test))

#------------------------------------------------------------------------------------------
# 2. 토큰화, 정제, 단어 사전 만들기
#------------------------------------------------------------------------------------------

# 토큰화
tokenizer = Okt()
train_tokens = [tokenizer.morphs(review) for review in train.text]
test_tokens = [tokenizer.morphs(review) for review in test.text]

# 단어 사전 만들기
vocab = build_vocab(corpus=train_tokens, n_vocab=5000, special_tokens=["<pad>", "<unk>"])
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}
print(vocab[:10])
print(len(vocab))

unk_id = token_to_id["<unk>"]
train_ids = [
    [token_to_id.get(token, unk_id) for token in review] for review in train_tokens
    ]
test_ids = [
    [token_to_id.get(token, unk_id) for token in review] for review in test_tokens
    ]

# 문장 길이 통일화
max_length = 32
pad_id = token_to_id['<pad>']

# 정수 인코딩 및 패딩 진행
train_ids = pad_sequences(train_ids, max_length, pad_id)
test_ids = pad_sequences(test_ids, max_length, pad_id)
print(train_ids[0])
print(test_ids[0])

#------------------------------------------------------------------------------------------
# 3. 데이터셋 만들기
#------------------------------------------------------------------------------------------

# 조건 지정
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 5
interval = 500

# 텐서화
train_ids = torch.tensor(train_ids)
test_ids = torch.tensor(test_ids)
train_lables = torch.tensor(train.label.values, dtype=torch.float32)
test_lables = torch.tensor(test.label.values, dtype=torch.float32)

# 데이터셋 만들기
train_dataset = TensorDataset(train_ids, train_lables)
test_dataset = TensorDataset(test_ids, test_lables)

# 데이터로더 적용
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 손실 함수와 최적화 함수 정의
n_vocab = len(token_to_id)
hidden_dim = 64
embedding_dim = 128
n_layers = 2

classifier = SentenceClassifier(
    n_vocab=n_vocab, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers
).to(device)

# 손실 함수 지정
criterion = nn.BCEWithLogitsLoss().to(device)

# 최적화 함수 지정
optimizer = optim.RMSprop(classifier.parameters(), lr=0.001)

# 학습 진행
for epoch in range(epochs):
    trainModel(classifier, train_loader, criterion, optimizer, device, interval)
    testModel(classifier, test_loader, criterion, device)

#------------------------------------------------------------------------------------------
# (+추가+) 학습된 모델로부터 임베딩 추출 (전이학습)
#------------------------------------------------------------------------------------------

token_to_embedding = dict()
embedding_matrix = classifier.embedding.weight.detach().cpu().numpy()

for word, emb in zip(vocab, embedding_matrix):
    token_to_embedding[word] = emb

token = vocab[1000]
print(token, token_to_embedding[token])

# 사전 학습된 모델로 임베딩 계층 초기화
word2vec = Word2Vec.load('../models/word2vec.model')
init_embeddings = np.zeros((n_vocab, embedding_dim))

for index, token in id_to_token.items():
    if token not in ['<pad>','<unk>']:
        init_embeddings[index] = word2vec.wv[token]

embedding_layer = nn.Embedding.from_pretrained(
    torch.tensor(init_embeddings, dtype=torch.float32)
)

# 조건2 (사전 학습된 임베딩을 사용한 모델 학습)
classifier2 = SentenceClassifier2(
    n_vocab=n_vocab, hidden_dim=hidden_dim, embedding_dim=embedding_dim,
    n_layers=n_layers, pretrained_embedding=init_embeddings
).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.RMSprop(classifier2.parameters(), lr=0.001)
epochs2 = 5
interval = 500

for epoch in range(epochs2):
    trainModel(classifier2, train_loader, criterion, optimizer, device, interval)
    testModel(classifier2, test_loader, criterion, device)