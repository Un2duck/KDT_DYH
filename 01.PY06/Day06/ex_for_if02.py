# ------------------------------------------------------------------------------
# 제어문 - 반복문과 조건문 혼합
# ------------------------------------------------------------------------------
## [실습]-----------------------------------------------------------------------
## 메세지를 입력 받습니다.
## 알파벳 대문자인경우 소문자로, 소문자인경우 대문자로
## 나머지는 그대로 되도록 출력하기
## -----------------------------------------------------------------------------
## 문자 ==> 코드 : ord(문자1개)
## 코드 ==> 문자 : chr(정수코드값)

msg=input('메세지 입력.:')
check=0
msg2=""
for m in msg:
    # 알파벳 대문자 'A'<=m<='Z'
    if ('A'<=m<='Z'):
        check=ord(m)+32
        print(chr(check), end='')
        msg2=msg2+chr(check)

    # 알파벳 소문자 'a'<=m<='z'
    elif ('a'<=m<='z'):
        check=ord(m)-32
        print(chr(check), end='')
        msg2=msg2+chr(check)
    else:
        print(m, end='')
        msg2=msg2+m
print(f'msg2 ==> {msg2}')


# 대문자 ---> 소문자
print(chr(ord('A')+32))

# 소문자 ---> 대문자
print(chr(ord('A')-32))