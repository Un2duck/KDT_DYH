# ---------------------------------------------------------------------
# Version.1
# file_name : DL_train_test(wine).py
# Date : 2024-09-18
# 설명 : wine.csv train val test 진행
# ---------------------------------------------------------------------
# 클래스 모듈 로딩 (DL_Reg_Class.py) 
# --------------------------------------------------------------------- 
from DL_Reg_Class import *

from makeaplot import loss_score_plot

# --------------------------------------------------------------------- 
# 데이터 준비
# ---------------------------------------------------------------------

DATA_FILE='../MyClass/Data/wine.csv'

# ---------------------------------------------------------------------
# 데이터 전처리
# ---------------------------------------------------------------------

# 스위치 삽입
switch = int(input('어떤 모델을 사용? (회귀: 0, 이진분류: 1)'))

# CSV => DataFrame
if switch == 0:
    # 회귀 시
    dataDF = pd.read_csv(DATA_FILE)
else:
    # 이진분류
    dataDF = pd.read_csv(DATA_FILE)

# ---------------------------------------------------------------------
# 데이터셋 인스턴스 생성
# DataFrame에서 피쳐와 타겟 추출
featureDF = dataDF[dataDF.columns[:-1]] # 2D (150, 3)
targetDF = dataDF[dataDF.columns[-1:]] # 2D (150, 1)

# - 커스텀데이터셋 인스턴스 생성
dataDS=MyDataset(featureDF, targetDF)
# ---------------------------------------------------------------------
# 학습 준비
# 하이퍼 파라미터 설정

EPOCHS = int(input('설정할 에포크를 입력.(ex.1000):'))
BATCH_SIZE = int(input('설정할 배치사이즈를 입력.(ex.10):'))
LR = float(input('설정할 LR값을 입력.(ex.0.001):'))

BATCH_CNT = dataDF.shape[0]//BATCH_SIZE
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'BATCH_CNT: {BATCH_CNT}')

# 모델 인스턴스 생성
if switch == 0: model=MyRegModel(3)
# if switch == 0: model=KeyDynamicModel(3, 10, 1, 30, 20, 10)
elif switch == 1: model=MyBCFModel()
else: model=MyMCFModel()

# model=KeyDynamicModel(4, 10, 1, 30, 20, 10)

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

# ---------------------------------------------------------------------
# 함수 이름 : training
# 함수 역할 : 배치 크기 만큼 데이터 로딩해서 학습 진행
# 매개 변수 : score_func
# ---------------------------------------------------------------------

def training():
    # 학습 모드로 모델 설정
    model.train()
    # 배치 크기 만큼 데이터 로딩해서 학습 진행
    loss_total, score_total=0,0
    for featureTS, targetTS in trainDL:

        # 학습 진행
        pre_y=model(featureTS)

        # 손실 계산
        loss=loss_func(pre_y, targetTS)
        loss_total+=loss.item()
        
        # 성능평가 계산
        score=score_func(pre_y, targetTS)
        score_total+=score.item()

        # 최적화 진행
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_total, score_total
# ---------------------------------------------------------------------
# 함수 이름 : validate
# 함수 역할 : 배치 크기 만큼 데이터 로딩해서 검증 진행
# 매개 변수 : loss_func, score_func
# ---------------------------------------------------------------------

def validate():
    # 검증 모드로 모델 설정
    model.eval()
    with torch.no_grad():
        # 검증 데이터셋
        val_featureTS=torch.FloatTensor(valDS.featureDF.values)
        val_targetTS=torch.FloatTensor(valDS.targetDF.values)
        
        # 평가
        pre_val=model(val_featureTS)

        # 손실
        loss_val=loss_func(pre_val, val_targetTS)

        # 성능평가
        score_val=score_func(pre_val, val_targetTS)
    return loss_val, score_val
    
# ---------------------------------------------------------------------
# 학습 효과 확인 => 손실값과 성능평가값 저장 필요
# ---------------------------------------------------------------------

loss_history, score_history=[[],[]], [[],[]]
print('TRAIN, VAL 진행')

start_time = time.time()
for epoch in range(1, EPOCHS):
    # 학습 모드 함수 호출
    loss_total, score_total = training()

    # 검증 모드 함수 호출
    loss_val, score_val = validate()

    # 에포크당 손실값과 성능평가값 저장
    loss_history[0].append(loss_total/epoch)
    score_history[0].append(score_total/epoch)

    loss_history[1].append(loss_val)
    score_history[1].append(score_val)

    print(f'{epoch}/{EPOCHS} => [TRAIN] LOSS: {loss_history[0][-1]} SCORE: {score_history[0][-1]}')
    print(f'\t=> [VAL] LOSS: {loss_history[1][-1]} SCORE: {score_history[1][-1]}')
end_time = time.time()
print(f'소요시간: {end_time-start_time:.3f}sec')

# ---------------------------------------------------------------------
# 테스트 진행
# ---------------------------------------------------------------------
print('TEST 진행')

model.eval()
with torch.no_grad():
    # 테스트 데이터셋
    test_featureTS=torch.FloatTensor(testDS.featureDF.values)
    test_targetTS=torch.FloatTensor(testDS.targetDF.values)

    # 평가
    pre_test=model(test_featureTS)

    # 손실
    loss_test=loss_func(pre_test, test_targetTS)

    # 성능평가
    score_test=score_func(pre_test, test_targetTS)
print(f'[TEST] LOSS: {loss_test} \n\tSCORE: {score_test}')

# ---------------------------------------------------------------------
# 함수 이름 : loss_score_plot
# 매개 변수 : loss, score, threshold=10 (default)
# 함수 역할 : 학습 후 loss, score 시각화 진행
# ---------------------------------------------------------------------

loss_score_plot(loss_history, score_history)