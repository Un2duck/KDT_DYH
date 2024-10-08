# ---------------------------------------------------------------------
# 모델링 관련 모듈 로딩
# ---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchinfo import summary

from torchmetrics.regression import R2Score, MeanSquaredError
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassConfusionMatrix

import torchvision.models as models

# ---------------------------------------------------------------------
# 데이터 분석 관련 모듈 로딩
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 이미지 관련 모듈 로딩
# ---------------------------------------------------------------------
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

# ---------------------------------------------------------------------
# 기타 모듈 로딩
# ---------------------------------------------------------------------
import time
import os

# ---------------------------------------------------------------------
# 커스텀 모듈 로딩
# ---------------------------------------------------------------------
import OpenCV_func

# 활용 패키지 버전 체크
print(f'torch Ver.:{torch.__version__}')
print(f'pandas Ver.:{pd.__version__}')
print(f'numpy Ver.:{np.__version__}')
# DEVICE 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

###  이미지 파일 => 하나로 합치기
## 이미지 파일 => 하나로 합치기
## 데이터 관련 설정
IMG_PATH = './test_image/'

### Tensor ==> Ndarray
## 데이터 변형 및 전처리
transConvert = v2.Compose([
    v2.Resize([256, 256]),
    v2.RandomResizedCrop(224),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.ToDtype(torch.float32, scale=True)
])

## 이미지 데이터셋 생성
imgDS=ImageFolder(root=IMG_PATH, transform=transConvert)
print(f'imgDS2.classes       : {imgDS.classes}')
print(f'imgDS2.class_to_idx  : {imgDS.class_to_idx}')
print(f'imgDS2.targets       : {imgDS.targets}')

# for img in imgDS.imgs:
#     print(f'imgDS.imgs       : {img}')

imgDL=DataLoader(imgDS, batch_size=32)

### 사전학습된 모델 로딩
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

### 사전학습된 모델의 파라미터 비활성화 설정
for named, param in model.named_parameters():
    # 역전파 시에 업데이트 되지 않도록 설정
    param.requires_grad=False

model.classifier[6] = nn.Linear(4096, 4)

### classifier 파라미터 활성화 설정
for named, param in model.classifier[6].named_parameters():
    param.requires_grad=True

model=model.to(DEVICE)

# 최적화 인스턴스
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# 손실함수 인스턴스
loss_func = torch.nn.CrossEntropyLoss()

since = time.time()    
acc_history = []
loss_history = []
best_acc = 0.0
num_epochs = 5
dataloader = imgDL

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)

    running_corrects = 0
    running_loss = 0.0

    for featureTS, targetTS in dataloader:
        
        # DEVICE : CPU/GPU 사용
        featureTS=featureTS.to(DEVICE)
        targetTS=targetTS.to(DEVICE)
        
        # 학습 진행
        pre_y = model(featureTS)

        # 손실 계산
        loss = loss_func(pre_y, targetTS)

        _, preds = torch.max(pre_y, 1)
        
        # 최적화 진행
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_corrects += torch.sum(preds == targetTS.data)
        running_loss += loss.item() * featureTS.size(0)

    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_loss = running_loss / len(dataloader.dataset)

    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    if epoch_acc > best_acc:
        best_acc = epoch_acc

    acc_history.append(epoch_acc.item())
    loss_history.append(epoch_loss)        

    ### 모델 파라미터 저장
    # torch.save(model.state_dict(), os.path.join('./models/', '{0:0=2d}.pth'.format(epoch)))
    print()

    ### 모델 저장
    # 끝나는 시간 저장
    end_time = time.strftime('%y.%m.%d..%H_%M_%S')

    # 모델 경로 지정
    SAVE_PATH = '../Project/models'
    SAVE_MODEL = f'/model_num_{end_time}'
    # torch.save(model, SAVE_PATH+SAVE_MODEL)

time_elapsed = time.time() - since
print('모델 학습 시간: {:.0f}분 {:.0f}초'.format(time_elapsed // 60, time_elapsed % 60))
print('Best Acc: {:4f}'.format(best_acc))

# OpenCV_func.loss_score_plot(loss_history, acc_history, threshold=5)