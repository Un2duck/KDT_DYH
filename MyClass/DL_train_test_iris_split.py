# ---------------------------------------------------------------------
# Version.1
# file_name : DL_train_test_iris_split.py
# Date : 2024-09-19
# 설명 : iris.csv train val test 진행
# 변경사항 : 모델 저장 추가
# ---------------------------------------------------------------------
# 클래스 모듈 로딩 (DL_Reg_Class.py) 
# --------------------------------------------------------------------- 
from DL_Reg_Class import *
from DL_Modeling import *
from DL_func import *
import os
from makeaplot import loss_score_plot

# --------------------------------------------------------------------- 
# 데이터 준비
# ---------------------------------------------------------------------

DATA_FILE='./Data/iris.csv'

# 저장 경로
SAVE_PATH='../Models/iris/MCF/'

# 저장 파일명
SAVE_FILE=SAVE_PATH+'MCFmodel_train_wbs.pth'

# 모델 구조 및 파라미터 모두 저장 파일명
SAVE_MODEL=SAVE_PATH+'MCFmodel_all.pth'

# 경로상 폴더 존재 여부 체크
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)  # 폴더 / 폴더 / ... 하위 폴더까지 생성

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

model, trainDL, trainDS, valDS, testDS, loss_func, score_func, optimizer = modeling(featureDF, targetDF, dataDF, switch)

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
    loss_total, score_total = training(model, trainDL, loss_func, score_func, optimizer)

    # 검증 모드 함수 호출
    loss_val, score_val = validate(model, loss_func, score_func, valDS)

    # 에포크당 손실값과 성능평가값 저장
    loss_history[0].append(loss_total/epoch)
    score_history[0].append(score_total/epoch)

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