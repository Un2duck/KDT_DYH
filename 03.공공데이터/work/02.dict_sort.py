'''
1. 딕셔너리 생성 및 정렬 프로그램
 - 제출 파일: dict_sort.py
아래에 주어진 총 6개 나라의 수도에 대한 국가명, 대륙, 인구수를 표시한 테이블을 이용하여
딕셔너리를 작성하고, 아래 실행 화면과 같이 출력을 하는 프로그램을 작성하시오.
수도이름(key) 국가명 대륙 인구수
Seoul South	Korea Asia 9,655,000
Tokyo Japan Asia 14,110,000
Beijing China Asia 21,540,000
London United Kingdom Europe 14,800,000
Berlin Germany Europe 3,426,000
Mexico	City Mexico America 21,200,000

[기능 구현 내용]
1. 전체 데이터 출력
 위의 테이블을 이용하여 dictionary를 생성하고 전체 데이터를 화면에 출력
 Key: 수도 이름, value: 국가명, 대륙, 인구수

2. 수도 이름 오름차순 출력
 수도 이름을 기준으로 오름차순 정렬한 다음 dictionary의 모든 데이터를 출력
 자리 수 맞춤

3. 모든 도시의 인구수 내림차순 출력
 인구수를 내림차순으로 정렬한 다음 수도이름, 인구수만 화면에 출력
 자리 수 맞춤

4. 특정 도시의 정보 출력
 화면에서 입력 받은 수도 이름이 딕셔너리의 key에 존재하면, 해당 수도의 모
든 정보를 화면에 출력함
 수도 이름이 딕셔너리에 존재하지 않으면 “도시이름:XXX은 key에 없습니다.”를 출력

5. 대륙별 인구수 계산 및 출력
 화면에서 대륙 이름을 입력 받고 해당 대륙에 속한 국가들의 인구수를 출력하고
 전체 인구수의 합을 계산하여 출력
 잘못된 대륙 이름 검사는 없음

6. 프로그램 종료
'''

capital = {'Seoul':['South Korea','Asia',9655000],
           'Tokyo':['Japan','Asia',14110000],
           'Beijing':['China','Asia',21540000],
           'London':['United Kingdom','Europe',14800000],
           'Berlin':['Germany','Europe',3426000],
           'Mexico City':['Mexico','America',21200000]}

# -------------------------------------------------------------------------------
# 초기 화면 (함수)

def monitor():
    print('----------------------------------------- ')
    print('1. 전체 데이터 출력')
    print('2. 수도 이름 오름차순 출력')
    print('3. 모든 도시의 인구수 내림차순 출력')
    print('4. 특정 도시의 정보 출력')
    print('5. 대륙별 인구수 계산 및 출력')
    print('6. 프로그램 종료')
    print('----------------------------------------- ')

# 1. 전체 데이터 출력 (함수)
def first():
    for k,v in capital.items():
        print(f'{k}: {v}')

# 2. 수도 이름 오름차순 출력
def second():
    sorted_capital = sorted(capital.items())
    for i in range(6):
        print(f'[{i+1}] {sorted_capital[i][0]}: {sorted_capital[i][1]}')

# 3. 모든 도시의 인구수 내림차순 출력
def third():
    x=sorted(capital.items(), key = lambda x:x[1][2], reverse=True)
    for k,v in x:
        print(f'{k:2}: {v[2]:2,}')

# 4. 특정 도시의 정보 출력
def fourth():
    menu2 = input('출력할 도시 이름을 입력하세요:')
    if menu2 in capital.keys():
        print(f'도시:{menu2}\n국가:{capital[menu2][0]}, 대륙:{capital[menu2][1]}, 인구수: {capital[menu2][2]}')
    else:
        print(f'도시이름: {menu2}은/는 key에 없습니다.')

# 5. 대륙별 인구수 계산 및 출력

def fifth():
    menu3 = input('대륙 이름을 입력하세요(Asia, Europe, America):')
    total = 0
    for k,v in capital.items():
        if v[1] == menu3:
            print(f'{k}: {v[2]:,}')
            total+=v[2]
    print(f'{menu3} 전체 인구수: {total:,}')

# -------------------------------------------------------------------------------

monitor()
menu = int(input('메뉴를 입력하세요.'))

while True:
    if menu == 6:
        break
    elif menu == 1:
        first()
        monitor()
        menu = int(input('메뉴를 입력하세요.'))
    elif menu == 2:
        second()
        monitor()
        menu = int(input('메뉴를 입력하세요.'))
    elif menu == 3:
        third()
        monitor()
        menu = int(input('메뉴를 입력하세요.'))
    elif menu == 4:
        fourth()
        monitor()
        menu = int(input('메뉴를 입력하세요.'))
    elif menu == 5:
        fifth()
        monitor()
        menu = int(input('메뉴를 입력하세요.'))
    else:
        monitor()
        menu = int(input('메뉴를 입력하세요.'))