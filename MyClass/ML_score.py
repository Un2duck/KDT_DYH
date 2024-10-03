# ---------------------------------------------------------------------
# Version.1
# file_name : ML_score.py
# Date : 2024-09-18
# 설명 : 머신러닝 score 계산 관련 class
# ---------------------------------------------------------------------
# 데이터 관련 모듈 로딩
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 스코어 관련 모듈 로딩
# ---------------------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_curve
from sklearn.metrics import precision_recall_curve

# ---------------------------------------------------------------------
# 함수 이름 : get_clf_eval
# 함수 역할 : 배치 크기 만큼 데이터 로딩해서 학습 진행
# 매개 변수 : y_test, pred
# ---------------------------------------------------------------------

def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print('오차 행렬: \n', confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}'.format(accuracy, precision))
    print('재현율: {0:.4f}'.format(recall))

def get_pred_proba(model, X_test):
    pred_proba = model.predict_proba(X_test)
    pred_proba_1 = pred_proba[:,1].reshape(-1,1)

def roc_curve_plot(y_test, pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음.
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)

def precision_recall_curve_plot(y_test, pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 
    # 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()