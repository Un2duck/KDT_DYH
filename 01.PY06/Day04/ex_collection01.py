# ------------------------------------------------------------------------------
# Collection 자료형에 공통적인 부분 살펴보기
# ------------------------------------------------------------------------------
## [여러개의 변수에 데이터 저장]--------------------------------------------------
# name='홍길동'
# age=12
# job='의적'
# gender='남'

# 패킹(Packing) 방식 : 변수명 = tuple 타입
data='홍길동', 12, '의적', '남'

# ------------------------------------------------------------------------------
# 언패킹(Unpacking) 방식 : 변수1, 변수2, 변수3, ... ,변수n = tuple 타입
# 변수명 개수 == 데이터 수
# ------------------------------------------------------------------------------

name, age, job, gender = '홍길동', 12, '의적', '남'
print(name, age, job, gender)

# name, age = '홍길동', 12, '의적', '남'
# # ValueError: too many values to unpack (expected 2)

name, _, job, _ = '마징가', 155, '의적', '남'
print(name, job, _)

score=[100,99]
kor, math = [100, 99]
print(score, kor, math)

person={'name':'박','age':12}
k1, k2={'name':'박','age':12}
print(person, k1, k2)

# ------------------------------------------------------------------------------
# 생성자(Constructor) 함수 : 타입명과 동일한 이름의 함수
# - int(), float(), str(), bool(), list(), tuple(), dict(), set()
# - map(), range()
# ------------------------------------------------------------------------------
# 기본 데이터 타입

num=int(10)         # num=10
fnum=float(10.2)    # fnum=10.2
msg=str('Good')     # msg='Good'
isOk=bool(False)    # isOk=False

# 컬렉션 데이터 타입
lnums=list([1, 2, 3, 4])        # lnums=[1, 2, 3, 4]
tnums=tuple((3,6,9))            # tnums=(3,6,9)
dnums=dict({'d1':10, 'd2':30})  # dnums={'d1':10, 'd2':30}
snums=set({1,1,3,3,5})          # snums={1,1,3,3,5}

print(lnums, tnums, dnums, snums)

# 타입 변경 => 형변환------------------------------------------------------------
# dict 자료형은 다른 자료형과 달리 데이터 형태가 다름
# - 데이터 형태 => 키:값
# - dict(키1=값1, 키2=값2, ...) : 키는 str만 가능. '',"" 사용불가

ds=dict(n1=1,n2=2,gender='남')
print(ds)

ds=dict([['name','마징가'],['age',12]])
print(ds)

# 내장함수 : zip() ==> 같은 인덱스의 요소끼리 묶어줌
l1=['name','age','gender']
l2=['마징가',12,'남']
l3=[False, True, True]
print(list(zip(l1,l2,l3)))

ds=dict(zip(l1,l2))
print(ds)