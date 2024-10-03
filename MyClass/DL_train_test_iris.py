# ---------------------------------------------------------------------
# Version.1
# file_name : DL_train_test(iris).py
# Date : 2024-09-19
# 설명 : iris.csv train val test 진행
# 변경사항 : 모델 저장 추가
# ---------------------------------------------------------------------
# 클래스 모듈 로딩 (DL_Reg_Class.py) 
# --------------------------------------------------------------------- 
from DL_Reg_Class import *
import os
from DL_func import *
from makeaplot import loss_score_plot

# --------------------------------------------------------------------- 
# 데이터 준비
# ---------------------------------------------------------------------

DATA_FILE='../Data/iris.csv'

# 저장 경로
SAVE_PATH='../Models/iris/MCF/'

# 저장 파일명
SAVE_FILE=SAVE_PATH+'MCFmodel_train_wbs.pth'

# 모델 구조 및 파라미터 모두 저장 파일명
SAVE_MODEL=SAVE_PATH+'MCFmodel_all.pth'

# 경로상 폴더 존재 여부 체크
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)  # 폴더 / 폴더 / ... 하위 폴더까지 생성

# ---------------------------------------------------------------------
# 데이터 전처리
# ---------------------------------------------------------------------

# 스위치 삽입
switch = int(input('어떤 모델을 사용? (회귀: 0, 이진분류: 1, 다중분류: 2)'))

# CSV => DataFrame
if switch == 0:
    # 회귀 시
    dataDF = pd.read_csv(DATA_FILE, usecols=[0,1,2,3])
else:
    # 이진분류, 다중분류 시
    dataDF = pd.read_csv(DATA_FILE)


if switch == 1:
    # 이진분류 시
    dataDF['variety'] = (dataDF['variety'] == 'Setosa')
    dataDF['variety']=dataDF['variety'].astype('int')

elif switch == 2:
    # 다중분류 시
    labels=dict(zip(dataDF['variety'].unique().tolist(),range(3)))
    print(f'labels => {labels}')

    dataDF['variety']=dataDF['variety'].replace(labels)

else:
    pass

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

EPOCHS = 1000
BATCH_SIZE = 10
LR = 0.01

# EPOCHS = int(input('설정할 에포크를 입력.(ex.1000):'))
# BATCH_SIZE = int(input('설정할 배치사이즈를 입력.(ex.10):'))
# LR = float(input('설정할 LR값을 입력.(ex.0.001):'))

BATCH_CNT = dataDF.shape[0]//BATCH_SIZE
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'BATCH_CNT: {BATCH_CNT}')

# 모델 인스턴스 생성
if switch == 0: model=MyRegModel(3)
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
print(f'[y_train(value_counts()): {y_train.value_counts()/y_train.shape[0]}, y_test(value_counts()): {y_test.value_counts()/y_test.shape[0]}, y_val(value_counts()): {y_val.value_counts()/y_val.shape[0]}')

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

BREAK_CNT = 0
THESHOLD = 9
start_time = time.time()

for epoch in range(1, EPOCHS):
    # 학습 모드 함수 호출
    loss_total, score_total = training()

    # 검증 모드 함수 호출
    loss_val, score_val = validate()

    # 에포크당 손실값과 성능평가값 저장
    loss_history[0].append(loss_total/len(trainDL))
    score_history[0].append(score_total/len(trainDL))

    loss_history[1].append(loss_val)
    score_history[1].append(score_val)

    # 학습 진행 모니터링/스케쥴링: 검증 DS 기준
    
    # Loss 기준
    if len(loss_history[1]) >= 2:
        if loss_history[1][-1] >= loss_history[1][-2]: BREAK_CNT += 1

    # # score 기준
    # if len(score_history[1]) >= 2:
    #     if score_history[1][-1] <= score_history[1][-2]: BREAK_CNT += 1

    # 성능이 좋은 학습 가중치 저장
    # SAVE_FILE=f'model_train_wbs_{epoch}_{score_val:3f}.pth'

    if len(score_history[1]) == 1:
        # 첫번째라서 무조건 모델 파라미터 저장
        torch.save(model.state_dict(), SAVE_FILE)
        # 모델 전체 저장
        torch.save(model, SAVE_MODEL)
    else:
        if score_history[1][-1] > max(score_history[1][:-1]):
            torch.save(model.state_dict(), SAVE_FILE)
            torch.save(model, SAVE_FILE)

    # 학습 중단 여부 설정
    if BREAK_CNT >= THESHOLD:
        print('성능 및 손실 개선이 없어서 학습 중단')
        break

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


# loss_score_plot(loss_history, score_history)

save_model(model)
