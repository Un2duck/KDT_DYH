'''
(윷놀이 게임 프로그램)

흥부와 놀부가 윷놀이를 하는데, 각자 4개의 윷가락을 교대로 던져서 20점 이상의 점수가 먼저 나오는 사람이 승리를 한다.
윷가락을 던져 나온 점수가 "윷(4점)"이나 "모(5점)"가 나오는 경우, 동일한 사람이 한 번 더 던질 수 있다.
아래 주어진 메소드를 구현하고 필요한 기능들은 추가로 구현하여 프로그램을 작성하시오.

배열의 값: 1111 (모, 점수: 5점)
배열의 값: 0000 (윷, 점수: 4점)
배열의 값: 1000 (걸, 점수: 3점)
배열의 값: 1100 (개, 점수: 2점)
배열의 값: 1110 (도, 점수: 1점)
'''

# # hint)
# while True:

#     while True:
#         흥부 윷던지기
#         if 조건:
#             break
#     if 조건: # 20점 이상
#         break
#         while True: # 놀부


# -------------------------------------------------------------------------------
# 모듈 생성
import random
# -------------------------------------------------------------------------------
# 윷 만들기
sticks=[0, 0, 0, 0]
names=['흥부','놀부']
# -------------------------------------------------------------------------------
# 함수 만들기
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
### 최상/최악의 경우 만들기 (함수)
# def throw():
#     sticks = 0
# -------------------------------------------------------------------------------

# 1. 윷 던지기 (함수)
# 0: 뒷면, 1: 앞면
def throw():
    for i in range(4):
        sticks[i] = random.randint(0, 1)

# 2. 윷 점수 (함수)
def sticks_score():
    sum_sticks = sum(sticks)
    sticks_name = ['도','개','걸','윷','모']
    if sum_sticks == 4: score=5 # 모
    elif sum_sticks == 0: score=4 # 윷
    elif sum_sticks == 1: score=3 # 걸
    elif sum_sticks == 2: score=2 # 개
    elif sum_sticks == 3: score=1 # 도
    return [sticks, score, sum_sticks, sticks_name[score-1]]

# 3. 흥부와 놀부 윷던지기 (함수)
def whos_throw(whos):
    throw()

    whos[0]+=sticks_score()[1]
    whos[1]+=1

# 4. 결과 나타내기: 윷 종류와 점수 (함수)
def print_throw(whos,number):

    # print(f'---{names[number]}{whos[1]}번째---')
    print(f'{names[number]} {sticks_score()[0]}: {sticks_score()[3]} ({sticks_score()[1]}점)/총 ({whos[0]})점')

# 5. 결과 나타내기: 누가 이겼는지 (함수)
def print_winner(whos,number):

    print(f'--- {names[number]} 최종스코어 :', whos[0])
    print(f'--- {names[number]} 최종횟수 :', whos[1])
    print(f'*** {names[number]} 승!!! ***')

# -------------------------------------------------------------------------------
# * (번외) 규칙 변경 윷 점수 함수 만들기 * - 개 나오면 => 0점 만들기

# # 1'.윷 던지기 (함수) - <동일>
# # 0: 뒷면, 1: 앞면
# def throw():
#     for i in range(4):
#         sticks[i] = random.randint(0, 1)

# # 2'.윷 점수 (함수) - <<변경>>
# def sticks_score():
#     sum_sticks = sum(sticks)
#     sticks_name = ['개','도','','걸','윷','모'] # score가 바뀜으로 리스트 순서 & 개수 바꿈.
#     if sum_sticks == 4: score=5 # 모
#     elif sum_sticks == 0: score=4 # 윷
#     elif sum_sticks == 1: score=3 # 걸
#     elif sum_sticks == 2: score=0 # 개 ## 2점 >> 0점
#     elif sum_sticks == 3: score=1 # 도
#     return [sticks, score, sum_sticks, sticks_name[score]]

# # 3'. 흥부와 놀부 윷던지기 (함수) - <<변경>>
# def whos_throw(whos):
#     throw()
#     if sticks_score()[2] != 2:
#         whos[0]+=sticks_score()[1]
#         whos[1]+=1
#     else:
#         whos[0]=0
#         whos[1]+=1

# # 4'. 결과 나타내기: 윷 종류와 점수 (함수) - <동일>
# def print_throw(whos,number):

#     print(f'---{names[number]}{whos[1]}번째---')
#     if number == 0: print('<---', end=" ")
#     print(f'{names[number]} {sticks_score()[0]}: {sticks_score()[3]} ({sticks_score()[1]}점)/총 ({whos[0]})점', end=" ")
#     if number == 1: print('--->')
# # 5'. 결과 나타내기: 누가 이겼는지 (함수) - <동일>
# def print_winner(whos,number):

#     print(f'--- {names[number]} 최종스코어 :', whos[0])
#     print(f'--- {names[number]} 최종횟수 :', whos[1])
#     print(f'*** {names[number]} 승!!! ***')

# -------------------------------------------------------------------------------
# 클래스 만들기
# -------------------------------------------------------------------------------

class Switch:

    # 힙 영역에 객체 생성 시 속성값 저장
    def __init__(self,whos,number):
        self.whos=whos
        self.number = number

    def whos_throw(self):
        throw()
        self.whos[0]+=sticks_score()[1]
        self.whos[1]+=1

    def print_throw(self):
        print(f'---{names[self.number]}{self.whos[1]}번째---')
        if self.number == 1: print('<---')
        print(f'{names[self.number]} {sticks_score()[0]}: {sticks_score()[3]} ({sticks_score()[1]}점)/총 ({self.whos[0]})점')
        if self.number == 0: print('--->')

    def print_winner(self):
        print(f'--- {names[self.number]} 최종스코어 :', self.whos[0])
        print(f'--- {names[self.number]} 최종횟수 :', self.whos[1])
        print(f'*** {names[self.number]} 승!!! ***')

# -------------------------------------------------------------------------------
hung = [0] * 2
# hung[0] : 흥부 윷 점수
# hung[1] : 흥부 윷 던진 횟수

nul = [0] * 2
# nul[0] : 놀부 윷 점수
# nul[1] : 놀부 윷 던진 횟수

# -------------------------------------------------------------------------------

# # 1. 흥부 혼자서 윷던지기

# while True:
#     if hung[0] >= 20:
#         print('최종스코어', hung[0])
#         print('최종횟수', hung[1])
#         break
#     else:
#         throw()
#         hung[0]+=sticks_score()[2]
#         hung[1]+=1
#         print(hung[0])
#         print(hung[1])

# -------------------------------------------------------------------------------

# 2. 흥부, 놀부 번갈아가면서 윷던지기

# while True:

#     # 흥부 점수가 20이 넘으면 빠져나감.
#     if hung[0] >= 20:
#         print_winner(hung,0)
#         break

#     # 놀부 점수가 20이 넘으면 빠져나감
#     elif nul[0] >= 20:
#         print_winner(nul,1)
#         break
#     else:
#         whos_throw(hung)
#         print_throw(hung,0)

#         whos_throw(nul)
#         print_throw(nul,1)

# -------------------------------------------------------------------------------

# 3. 흥부, 놀부 번갈아가되, 윷이나 모가 나오면 한번 더 던지게 하기.

# while True:

#     # 흥부 점수가 20점이 넘으면 빠져나감.
#     if hung[0] >= 20:
#         print_winner(hung,0)
#         break

#     # 놀부 점수가 20점이 넘으면 빠져나감
#     elif nul[0] >= 20:
#         print_winner(nul,1)
#         break

#     else:
#         if hung[0] < 20 and nul[0] < 20: # (추가. 두 사람 총점이 각각 20점 미만일 때만)
#             whos_throw(hung)
#             print_throw(hung,0)

#         # 흥부, '윷'이나 '모'나오면 한번 더 던짐.
#         while True:
#             if ((sticks_score()[1] == 4) or (sticks_score()[1] == 5)) and hung[0] < 20 and nul[0] < 20: 
#                 # (추가. 두 사람 총점이 각각 20점 미만일 때만)
#                 whos_throw(hung)
#                 print_throw(hung,0)
#             else:
#                 break

#         if hung[0] < 20 and nul[0] < 20: # (추가. 두 사람 총점이 각각 20점 미만일 때만)
#             whos_throw(nul)
#             print_throw(nul,1)

#         # 놀부, '윷'이나 '모'나오면 한번 더 던짐.
#         while True:
#             if ((sticks_score()[1] == 4) or (sticks_score()[1] == 5)) and hung[0] < 20 and nul[0] < 20: 
#                 # (추가. 두 사람 총점이 각각 20점 미만일 때만)
#                 whos_throw(nul)
#                 print_throw(nul,1)
#             else:
#                 break

# 4. 클래스 적용 버전

# 흥부 객체생성
hungObj = Switch(hung,0)

# 놀부 객체생성
nulObj = Switch(nul,1)

while True:

    # 흥부 점수가 20점이 넘으면 빠져나감.
    if hung[0] >= 20:
        hungObj.print_winner()
        break

    # 놀부 점수가 20점이 넘으면 빠져나감
    elif nul[0] >= 20:
        nulObj.print_winner()
        break

    else:
        if hung[0] < 20 and nul[0] < 20: # (추가. 두 사람 총점이 각각 20점 미만일 때만)
            hungObj.whos_throw()
            hungObj.print_throw()

        # 흥부, '윷'이나 '모'나오면 한번 더 던짐. (추가. 두 사람 총점이 각각 20점 미만일 때만)
        while True:
            if ((sticks_score()[1] == 4) or (sticks_score()[1] == 5)) and hung[0] < 20 and nul[0] < 20:
                hungObj.whos_throw()
                hungObj.print_throw()
            else:
                break

        if hung[0] < 20 and nul[0] < 20: # (추가. 두 사람 총점이 각각 20점 미만일 때만)
            nulObj.whos_throw()
            nulObj.print_throw()

        # 놀부, '윷'이나 '모'나오면 한번 더 던짐. (추가. 두 사람 총점이 20점 이하일 때만)
        while True:
            if ((sticks_score()[1] == 4) or (sticks_score()[1] == 5)) and hung[0] < 20 and nul[0] < 20:
                nulObj.whos_throw()
                nulObj.print_throw()
            else:
                break