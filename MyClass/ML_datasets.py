# ---------------------------------------------------------------------
# Version.1
# file_name : ML_datasets.py
# Date : 2024-09-16
# 설명 : 사이킷런 내장 데이터셋 탐색
# ---------------------------------------------------------------------
# ML 모델링 관련 모듈 로딩 (ML_Module.py)
# ---------------------------------------------------------------------
from ML_Module import *

# ---------------------------------------------------------------------
# 사용 데이터 준비 (사이킷런 데이터셋)
# ---------------------------------------------------------------------
from sklearn.datasets import *

# ---------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------
num = int(input('사용할 데이터를 선택하세요. (0. load_iris, 1. load_breast_cancer, 2. load_digit, 3. load_wine):'))

if num == 0: dataset = load_iris
elif num == 1: dataset = load_breast_cancer
elif num == 2: dataset = load_digits
elif num == 3: dataset = load_wine

# 피쳐 및 라벨(타겟) 선정
features = dataset().data # feature만으로 된 데이터 (numpy)
target = dataset().target # 레이블(결정 값) (numpy)

print('target:', target)
print('dataset.target_names:', dataset().target_names)

# dataset 데이터 -> dataset DataFrame화
iris_df = pd.DataFrame(data=features, columns=dataset().feature_names)
iris_df['label'] = dataset().target

# 학습용 데이터, 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=.2,
                                                    random_state=11)


# DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

# 학습 수행
dt_clf.fit(X_train, y_train)

# 학습 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측수행.
pred = dt_clf.predict(X_test)

# print(pred)
# print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))

# ---------------------------------------------------------------------
# 함수이름 : check_data()
# 함수목적 : 내장된 예제 데이터셋 정보확인
# 매개변수 : data_ (내장 데이터셋)
# ---------------------------------------------------------------------

def check_data(data_):
    load_data = data_()
    print(type(load_data))

    keys = load_data.keys()
    print(f'{data_.__name__} 데이터셋의 키들:{keys}')

    print('\n feature_names의 type:', type(load_data.feature_names))
    print(' feature_names의 shape:', len(load_data.feature_names))
    print(load_data.feature_names)

    print('\n target_names의 type:', type(load_data.target_names))
    print(' target_names의 shape:', len(load_data.target_names))
    print(load_data.target_names)

    print('\n data의 type:', type(load_data.data))
    print(' data의 shape:', load_data.data.shape)

    print('\n target의 type:', type(load_data.target))
    print(' target의 shape:', load_data.target.shape)
    print(load_data.target)
    return load_data

# ---------------------------------------------------------------------
# 함수이름 : cross_check()
# 함수목적 : 교차 검증
# 매개변수 :
# ---------------------------------------------------------------------

def cross_check():
    # 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성.
    kfold = KFold(n_splits=5)
    cv_accuracy = []
    print(f'{dataset.__name__} 데이터 세트 크기:', features.shape[0])

    n_iter = 0
    # KFold 객체의 split()를 호출하면 폴드별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환
    for train_index, test_index in kfold.split(features):
        
        # kfold.split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # 학습 및 예측
        dt_clf.fit(x_train, y_train)
        pred = dt_clf.predict(x_test)
        n_iter += 1

        # 반복 시마다 정확도 측정
        accuracy = np.round(accuracy_score(y_test, pred), 4)
        train_size = x_train.shape[0]
        test_size = x_test.shape[0]
        print('\n#{0} 교차 검증 정확도: {1}, 학습 데이터 크기: {2}, 검증 데이터 크키: {3}'
            .format(n_iter, accuracy, train_size, test_size))
        print('#{0} 검증 세트 인덱스: {1}'.format(n_iter, test_index))
        cv_accuracy.append(accuracy)

    # 개별 iteration별 정확도를 합하여 평균 정확도 계산
    print(f'\n## 평균 검증 정확도: {np.mean(cv_accuracy):.4f}')

# ---------------------------------------------------------------------
# 함수이름 : check_kfold
# 함수목적 : Kfold 교차 검증
# 매개변수 : X
# ---------------------------------------------------------------------

kfold = KFold(n_splits=3)

def check_kfold():
    n_iter = 0
    for train_index, test_index in kfold.split(iris_df):
        n_iter += 1
        label_train = iris_df['label'].iloc[train_index]
        label_test = iris_df['label'].iloc[test_index]
        print('## 교차 검증: {0}'.format(n_iter))
        print('학습 레이블 데이터 분포:\n', label_train.value_counts())
        print('검증 레이블 데이터 분포:\n', label_test.value_counts())

# ---------------------------------------------------------------------
# 함수이름 : check_stra_fold
# 함수목적 : Kfold 교차 검증
# 매개변수 : X
# ---------------------------------------------------------------------

skf = StratifiedKFold(n_splits=3)

def check_stra_fold():
    n_iter = 0
    for train_index, test_index in skf.split(features, iris_df['label']):
        n_iter += 1
        label_train = iris_df['label'].iloc[train_index]
        label_test = iris_df['label'].iloc[test_index]
        print('## 교차 검증: {0}'.format(n_iter))
        print('학습 레이블 데이터 분포:\n', label_train.value_counts())
        print('검증 레이블 데이터 분포:\n', label_test.value_counts())

# ---------------------------------------------------------------------
# 함수이름 : check_skfold
# 함수목적 : StratifiedKFold 교차 검증 후 정확도 확인
# 매개변수 : 
# ---------------------------------------------------------------------

def check_skfold():
    dt_clf = DecisionTreeClassifier(random_state=156)
    skfold = StratifiedKFold(n_splits=3)
    n_iter=0
    cv_accuracy=[]

    # StratifiedKFold의 split( ) 호출시 반드시 레이블 데이터 셋도 추가 입력 필요  
    for train_index, test_index  in skfold.split(features, target):
        # split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # 학습 및 예측 
        dt_clf.fit(X_train , y_train)    
        pred = dt_clf.predict(X_test)

        # 반복 시 마다 정확도 측정 
        n_iter += 1
        accuracy = np.round(accuracy_score(y_test, pred), 4)
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
        print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
              .format(n_iter, accuracy, train_size, test_size))
        print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
        cv_accuracy.append(accuracy)
        
    # 교차 검증별 정확도 및 평균 정확도 계산 
    print('\n## 교차 검증별 정확도:', np.round(cv_accuracy, 4))
    print('## 평균 검증 정확도:', np.round(np.mean(cv_accuracy), 4))

# ---------------------------------------------------------------------
# 함수이름 : check_cross_val_score
# 함수목적 : cross_val_score 확인
# 매개변수 : 
# ---------------------------------------------------------------------

def check_cross_val_score():
    features = load_iris()
    dt_clf = DecisionTreeClassifier(random_state=156)

    data = features.data
    label = features.target

    # 성능 지표는 정확도(accuracy) , 교차 검증 세트는 3개 
    scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=3)
    print('교차 검증별 정확도:', np.round(scores, 4))
    print('평균 검증 정확도:', np.round(np.mean(scores), 4))

# ---------------------------------------------------------------------
# 함수이름 : check_GridSearchCV
# 함수목적 : 교차 검증과 최적 하이퍼 파라미터 튜닝
# 매개변수 : 
# ---------------------------------------------------------------------

def check_GridSearchCV():
    # 데이터를 로딩하고 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        target, 
                                                        test_size=0.2, 
                                                        random_state=121)

    dtree = DecisionTreeClassifier()

    ### parameter 들을 dictionary 형태로 설정
    parameters = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}

    # param_grid의 하이퍼 파라미터들을 3개의 train, test set fold 로 나누어서 테스트 수행 설정.  
    ### refit=True 가 default 임. True이면 가장 좋은 파라미터 설정으로 재 학습 시킴.  
    grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)

    # 붓꽃 Train 데이터로 param_grid의 하이퍼 파라미터들을 순차적으로 학습/평가 .
    grid_dtree.fit(X_train, y_train)

    # GridSearchCV 결과 추출하여 DataFrame으로 변환
    scores_df = pd.DataFrame(grid_dtree.cv_results_)
    scores_df[['params', 'mean_test_score', 'rank_test_score', 
            'split0_test_score', 'split1_test_score', 'split2_test_score']]

    print('GridSearchCV 최적 파라미터:', grid_dtree.best_params_)
    print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dtree.best_score_))

    # GridSearchCV의 refit으로 이미 학습이 된 estimator 반환
    estimator = grid_dtree.best_estimator_

    # GridSearchCV의 best_estimator_는 이미 최적 하이퍼 파라미터로 학습이 됨
    pred = estimator.predict(X_test)
    print('테스트 데이터 세트 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))

# check_data(dataset)

def main():
    cross_check()
    check_kfold()
    check_stra_fold()
    check_GridSearchCV()

main()