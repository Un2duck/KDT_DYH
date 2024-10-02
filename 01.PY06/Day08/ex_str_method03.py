# -----------------------------------------------------------------------------------------------
# str 데이터 타입 전용 함수. 즉, 메서드 살펴보기
# -----------------------------------------------------------------------------------------------
##[문자열을 구성하는 문자를 검사해주는 메서드] --------------------------------------------------
## -isXXXX() ----> 결과 논리형 True/False
## [1] 알파벳으로 구성된 문자열인지 검사 isalpha()

data='good'
print(f'{data} => {data.isalpha()}')
print('-'*10)

## [2] 알파벳으로 구성된 문자열이 대문자인지 검사 : isupper()
data='good'
print(f'{data} => {data.isupper()}')
print(f'GOOD   => {"GOOD".isupper()}')
print(f'Good   => {"Good".isupper()}')
print('-'*10)

## [3] 알파벳으로 구성된 문자열이 소문자인지 검사 : islower()
print(f'GOOD => {data.islower()}')
print(f'Good   => {"Good".islower()}')
print(f'good   => {"good".islower()}')
print('-'*10)

## [4] 숫자로 구성된 문자열인지 검사 : isdecimal()
print(f'1234 => {"1234".isdecimal()}')
print(f'Happy1234   => {"Happy1234".isdecimal()}')
print('-'*10)

## [5] 숫자와 문자가 혼합된 문자열인지 검사 : isalnum()
print(f'1234 => {"1234".isalnum()}')
print(f'Happy1234   => {"Happy1234".isalnum()}')
print('-'*10)

## [6] 공백 문자 여부 검사 : isspace()
print(f'1234 => {"1234".isspace()}')
print(f'Happy1234   => {"Happy1234".isspace()}')
