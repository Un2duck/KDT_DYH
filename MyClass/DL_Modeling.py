# ---------------------------------------------------------------------
# Version.1
# file_name : DL_Modeling.py
# Date : 2024-09-19
# 설명 : train val test 진행용
# ---------------------------------------------------------------------
from DL_Reg_Class import *

# - 학습 준비
# 하이퍼 파라미터 설정

EPOCHS = 1000
BATCH_SIZE = 10
LR = 0.01

# EPOCHS = int(input('설정할 에포크를 입력.(ex.1000):'))
# BATCH_SIZE = int(input('설정할 배치사이즈를 입력.(ex.10):'))
# LR = float(input('설정할 LR값을 입력.(ex.0.001):'))


def modeling(featureDF, targetDF, dataDF, switch):

    # - 커스텀데이터셋 인스턴스 생성
    dataDS=MyDataset(featureDF, targetDF)
    
    BATCH_CNT = dataDF.shape[0]//BATCH_SIZE
    
    print(f'BATCH_CNT: {BATCH_CNT}')

    # 모델 인스턴스 생성
    if switch == 0: model=MyRegModel(4)
    # if switch == 0: model=KeyDynamicModel(3, 10, 1, 30, 20, 10)
    elif switch == 1: model=MyBCFModel()
    else: model=MyMCFModel()

    # model=KeyDynamicModel()
    # model=KeyDynamicModel(3, 10, 1, 30)

    # 데이터셋 인스턴스 생성
    X_train, X_test, y_train, y_test = train_test_split(featureDF, targetDF, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=1)

    print(f'[X_train(shape): {X_train.shape} (type): {type(X_train)}], X_test: {X_test.shape}, X_val: {X_val.shape}')
    print(f'[y_train(shape): {y_train.shape} (type): {type(y_train)}], y_test: {y_test.shape}, y_val: {y_val.shape}')

    trainDS=MyDataset(X_train, y_train)
    valDS=MyDataset(X_val, y_val)
    testDS=MyDataset(X_test, y_test)

    # 데이터로더 인스턴스 생성
    trainDL=DataLoader(trainDS, batch_size=BATCH_SIZE)

    # 최적화 인스턴스 생성
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 손실함수 인스턴스 생성
    if switch == 0: loss_func = nn.MSELoss()
    # 이진분류 BinaryCrossEntropyLoss => 예측값은 확률값으로 전달 ==> sigmoid() AF 처리 후 전달
    elif switch == 1: loss_func = nn.BCELoss()
    # 다중분류 CrossEntropyLoss => 예측값은 선형식 결과값 전달 ==> AF 처리 X
    else: loss_func = nn.CrossEntropyLoss()

    # 성능평가 함수
    if switch == 0: score_func = R2Score()
    elif switch == 1: score_func = BinaryF1Score()
    else: score_func = MulticlassF1Score(num_classes=3)
    
    return model, trainDL, trainDS, valDS, testDS, loss_func, score_func, optimizer