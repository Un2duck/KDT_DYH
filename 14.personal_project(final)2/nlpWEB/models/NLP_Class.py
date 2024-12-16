# 모듈 로딩
import torch
import torch.nn as nn
from torchinfo import summary
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# LSTM 모델 정의
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
    
def predict_LSTM(SAVE_MODEL, sample):
    NLP_model=torch.load(SAVE_MODEL, weights_only=False)

    # 저장된 Vectorizer 로드
    tfidf_vectorizer = joblib.load('../LSTM/tfidf_vectorizer.pkl')

    # 샘플 데이터 TF-IDF 벡터 변환
    sample_vectors = tfidf_vectorizer.transform(sample).toarray()

    # 3. 학습된 모델 로딩
    NLP_model.eval()

    # 추론 모드에서 그래디언트 비활성화
    with torch.no_grad():
        # 4. 모델 예측
        sample_tensor = torch.tensor(sample_vectors, dtype=torch.float32)
        predictions = NLP_model(sample_tensor).squeeze()

        # predicted_labels = predictions.argmax(dim=1)

        # predictions이 0차원 텐서인 경우 처리
        if predictions.ndim == 0:
            predictions = predictions.unsqueeze(0)

        # argmax 호출 전 차원 확인 및 처리
        if predictions.ndim == 1:
            predicted_labels = predictions.argmax(dim=0, keepdim=True)
        else:
            predicted_labels = predictions.argmax(dim=1)

    # 5. 결과 출력
    result_list = [[], []]
    
    for text, pred in zip(sample, predicted_labels):
        result = "어뷰징" if pred.item() == 0 else "정상"
        print(f"리뷰: {text}\n예측: {result}\n")
        result_list[0].append(text)
        result_list[1].append(result)
    return result_list