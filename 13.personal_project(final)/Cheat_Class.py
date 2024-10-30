# ---------------------------------------------------------------------
# Version.1
# file_name : Cheat_Class.py
# Date : 2024-10-28
# 설명 : Cheat Model 모델링용 모듈
# ---------------------------------------------------------------------

# 모델링 관련 모듈 로딩
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------------------------------------------------------------------
# 데이터셋 이름 : CheatDataset
# 부모클래스 : Dataset 
# 매개 변수 : 
# ---------------------------------------------------------------------
class CheatDataset(Dataset):

    def __init__(self, featureDF, targetDF):
        self.featureDF=featureDF
        self.targetDF=targetDF
        self.n_rows=featureDF.shape[0]
        self.n_features=featureDF.shape[1]

    def __len__(self):
        return self.n_rows
    
    def __getitem__(self, index):
        # 텐서화
        featureTS = torch.FloatTensor(self.featureDF.iloc[index])
        targetTS = torch.FloatTensor(self.targetDF.iloc[index])

        # 피쳐와 타겟 반환
        return featureTS, targetTS

# ---------------------------------------------------------------------
# 모델 이름 : MyCheatModel
# 부모클래스 : nn.Module 
# 매개 변수 : in_in, in_out, out_out, *hid
# ---------------------------------------------------------------------
class MyCheatModel(nn.Module):

    # 모델 구조 설계 함수 즉, 생성자 메서드
    def __init__(self, in_in, in_out, out_out, *hid):
        super().__init__()

        self.in_layer=nn.Linear(in_in, hid[0] if len(hid) else in_out)
        self.in_hids=nn.ModuleList()
        for i in range(len(hid)-1):
            self.in_hids.append(nn.Linear(hid[i], hid[i+1]))

        self.out_layer=nn.Linear(hid[-1] if len(hid) else in_out, out_out)

    # 학습 진행 콜백 메서드
    def forward(self, x):
        # 입력층
        y = F.relu(self.in_layer(x))

        # 은닉층
        for in_hid in self.in_hids:
            y = F.relu(in_hid(y))

        # 출력층
        return F.sigmoid(self.out_layer(y))