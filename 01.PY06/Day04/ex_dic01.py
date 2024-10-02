# ------------------------------------------------------------------------------
# Dict 자료형 살펴보기
# - 데이터의 의미를 함께 저장하는 자료형
# - 형태 : { 키1:값, 키2:값,...., 키n:값}
# - 키는 중복x, 값은 중복o
# - 데이터 분석 시 파일 데이터 가져 올 때 많이 사용
# ------------------------------------------------------------------------------

# [Dict 생성]-------------------------------------------------------------------
data={}
print(f'data => {len(data)}개, {type(data)}, {data}')

# 사람정보 : 이름, 나이, 성별
Data={'name':'마징가', 'age':100, 'gender':'남'}
print(f'data => {len(Data)}개, {type(Data)}, {Data}')

# 강아지정보 : 품종, 무게, 색상, 성별, 나이
dog={'type':'푸들','gram':5,'color':'brown','gender':'female','age':3}
print(f'data => {len(dog)}개, {type(dog)}, {dog}')

# 색상 출력
print(f'색상 : {dog["color"]}')

# 성별, 품종 출력
print(f'강아지 성별 : {dog["gender"]}, 강아지 종류: {dog["type"]}')

dog['age']=4
print(dog)

# 몸무게 5 => 7

dog['gram']=7
print(dog)

del dog['gender'] # del(dog['gender'])
print(dog)

# 추가 ==> 변수명[새로운 키]=값--------------------------------------
dog['name']='뽀삐'
print(dog)

dog['name']='마징가' # key가 있으면 업데이트
print(dog)