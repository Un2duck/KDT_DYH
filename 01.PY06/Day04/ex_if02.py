# ------------------------------------------------------------------------------
# [실습] 숫자를 입력 받아서 음이 아닌 정수와 음수 구분하기
# - 출력예시 : 숫자 -5는 음수입니다.
# ------------------------------------------------------------------------------

# num=int(input('숫자는 얼마입니까?'))
# if num > 0:
#     print(f'숫자 {num}은/는 음이 아닌 정수입니다.')
# elif num < 0:
#     print(f'숫자 {num}은/는 음수입니다.')
# else:
#     print('0은 제외합니다.')


# ------------------------------------------------------------------------------
# [실습] 점수를 입력 받아서 합격과 불합격 출력
# - 합격 : 60점 이상
# ------------------------------------------------------------------------------
level=['A','B','C','D','F']
score=int(input('점수는 몇 점입니까?'))

if score >= 60:
    print('합격')
else:
    print('불합격')

# ------------------------------------------------------------------------------
# [실습] 점수를 입력 받아서 학점 출력
# - 학점 : A (90>=), B(80>=), C(70>=), D(60>=), F
# ------------------------------------------------------------------------------

if score >= 90:
    print(f'학점은 {level[0]}입니다.')
elif score >= 80:
    print(f'학점은 {level[1]}입니다.')
elif score >= 70:
    print(f'학점은 {level[2]}입니다.')
elif score >= 60:
    print(f'학점은 {level[3]}입니다.')
else:
    print(f'학점은 {level[-1]}입니다.')