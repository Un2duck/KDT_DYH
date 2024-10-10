#------------------------------------------------------------------------------------------
# 모듈 로딩
#------------------------------------------------------------------------------------------

# 파이썬 내장 모듈
from collections import Counter

# 데이터분석 관련 모듈
import pandas as pd
import numpy as np

# Pytorch 모듈
import torch

#------------------------------------------------------------------------------------------
# 사용자 정의 함수
#------------------------------------------------------------------------------------------

# 데이터 토큰화 및 단어 사전 구축
def build_vocab(corpus, n_vocab, special_tokens):
    counter = Counter()
    for tokens in corpus:
        counter.update(tokens)
    vocab = special_tokens
    for token, count in counter.most_common(n_vocab):
        vocab.append(token)
    return vocab

# 정수 인코딩 및 패딩 진행 함수
def pad_sequences(sequences, max_length, pad_value):
    result = list()
    for sequence in sequences:
        sequence = sequence[:max_length]
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [pad_value] * pad_length
        result.append(padded_sequence)
    return np.asarray(result)

# 모델 학습 함수
def trainModel(model, datasets, criterion, optimizer, device, interval):
    model.train()
    losses = list()

    # for step, (input_ids, labels) in enumerate(datasets):
    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        # logits = model(input_ids)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % interval == 0:
            print(f'Train Loss {step}: {np.mean(losses)}')

# 모델 테스트 함수
def testModel(model, datasets, criterion, device):
    model.eval()
    losses = list()
    corrects = list()

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        yhat = torch.sigmoid(logits)>.5
        corrects.extend(
            torch.eq(yhat, labels).cpu().tolist()
        )
    print(f'Val Loss : {np.mean(losses)}, Val Accuracy : {np.mean(corrects)}')

#------------------------------------------------------------------------------------------
# 한국어 불용어 제거 함수
#------------------------------------------------------------------------------------------

# def remove_stopwords(tokens):
#     return [token for token in tokens if token not in stopwords]


#------------------------------------------------------------------------------------------
# 제너레이터 활용 함수
#------------------------------------------------------------------------------------------

def generateToken(dataset, load):
    for text, label in dataset:
        token_list = []
        doc = load(text)

        for token in doc:
            if (not token.is_punct) and (not token.is_stop):
                token_list.append(str(token))
        yield token_list

#------------------------------------------------------------------------------------------