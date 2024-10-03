# ---------------------------------------------------------------------
# Version.1
# file_name : ML_Module.py
# Date : 2024-09-18
# 설명 : ML 모델링 관련 모듈 로딩하기
# ---------------------------------------------------------------------
# ML 모델링 관련 모듈 로딩
# ---------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# 교차검증용 모듈
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------
# 데이터 관련 모듈 로딩
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 활용 패키지 버전 체크
# ---------------------------------------------------------------------

print(f'pandas Ver.:{pd.__version__}')
print(f'numpy Ver.:{np.__version__}')
print(f'seaborn Ver.:{sns.__version__}')
