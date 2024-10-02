# ------------------------------------------------------------------------------
# [실습1] 글자를 입력 받습니다.
#        입력받은 글자의 (a~z, A~Z) 코드값을 출력합니다.
# ------------------------------------------------------------------------------

data=input('글자를 입력하세요.(예시)a~z, A~Z):')

# 문자 ==> 코드값 변환 내장함수 : ord(문자 1개)
# if len(data)>0:
#     if len(data)==1:
#         if 'a'<=data<='z' or 'A'<=data<='Z':
#             print(f'{data}의 코드값: {ord(data)}')
#         else:
#             print("대소문자 알파벳만 입력하세요.")
#     else:
#         print("1개 문자만 입력하세요.")
# else:
#     print("입력한 데이터가 없습니다.")

if len(data) and ('a'<=data<='z' or 'A'<=data<='Z'):
    print(f'{data}의 코드값: {ord(data)}')
else:
    print("1개의 대소문자 알파벳만 입력하세요.\n입력한 데이터가 없습니다.")

data='ab'
print(list(map(ord, data)))

# ------------------------------------------------------------------------------
# [실습2] 점수를 입력 받은 후 학점을 출력합니다.
# - 학점: A+, A, A-, B+, B, B-, C+, C, C-, D+, D, D-, F
# A+ : 96 ~ 100
# A  : 95
# A- : 90 ~ 94
# ------------------------------------------------------------------------------

score=int(input('점수를 입력하세요.'))

if score >= 90:
    if score >= 96:
        print('A+')
    elif score == 95:
        print('A')
    else:
        print('A-')       
elif score >= 80:
    if score >= 86:
        print('B+')
    elif score == 85:
        print('B')
    else:
        print('B-')
elif score >= 70:
    if score >= 76:
        print('C+')
    elif score == 75:
        print('C')
    else:
        print('C-')
else:
    if score >= 66:
        print('D+')
    elif score == 65:
        print('D')
    else:
        print('D-')

jumsu=75
grade=""

if jumsu<0 or jumsu>100:
    print(f"{jumsu}는 잘못 입력된 점수입니다.")
else:
    if jumsu>95: grade='A+'
    elif jumsu==95: grade='A'
    elif jumsu>=90: grade='A-'
    elif jumsu>85: grade='B+'
    elif jumsu==85: grade='B'
    elif jumsu>=80: grade='B-'
    elif jumsu>75: grade='C+'
    elif jumsu==75: grade='C'
    elif jumsu>=70: grade='C-'
    elif jumsu>65: grade='D+'
    elif jumsu==65: grade='D'
    elif jumsu>=60: grade='D-'
    else: grade='F'
    print(f"{jumsu}는 {grade}학점입니다.")