# ---------------------------------------------------------------------
# Version.1
# file_name : NLP_Class.py
# Date : 2024-11-28
# 설명 : NLP 모델링용 모듈
# ---------------------------------------------------------------------

# 모델링 관련 모듈 로딩
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# 모델 이름 : LSTMClassifier
# 부모클래스 : nn.Module 
# 매개 변수 : input_dim
# 설명 : FNN(Feedforward Neural Network) 클래스 정의
# ---------------------------------------------------------------------

class AbuseClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AbuseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

# ---------------------------------------------------------------------
# 모델 이름 : LSTMClassifier
# 부모클래스 : nn.Module 
# 매개 변수 : input_dim, hidden_dim, output_dim, n_layers
# 설명 : LSTM 모델 정의
# ---------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out