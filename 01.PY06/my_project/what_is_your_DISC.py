# -----------------------------------------------------------------------------
# DISC 검사
# Dominant : 주도형
# Influential : 사교형
# Steady : 안정형
# Conscientious : 신중형
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
## 질문지
# -----------------------------------------------------------------------------

# 1. A.절제하는 B.강력한 C.꼼꼼한 D.표현력있는
# 2. A.개척적인 B.정확한 C.흥미진진한 D.만족스러운
# 3. A.기꺼이하는 B.활기있는 C.대담한 D.정교한
# 4. A.논쟁을 좋아하는 B.회의적인 C.주저하는 D.예측할 수 없는
# 5. A.공손한 B.사교적인 C.참을성이 있는 D.무서움을 모르는
# 6. A.설득력 있는 B.독립심이 강한 C.논리적인 D.온화한
# 7. A.신중한 B.차분한 C.과단성 있는 D.파티를 좋아하는
# 8. A.인기있는 B.고집있는 C.완벽주의자 D.인심 좋은
# 9. A.변화가 많은 B.수줍음을 타는 C.느긋한 D.완고한
# 10. A.체계적인 B.낙관적인 C.의지가 강한 D.친절한
# 11. A.엄격한 겸손한 상냥한 말주변이 좋은
# 12. A.호의적인 B.빈틈없는 C.놀기 좋아하는 D.의지가 강한
# 13. A.참신한 B.모험적인 C.절제된 D.신중한
# 14. A.참는 B.성실한 C.공격적인 D.매력있는
# 15. A.열정적인 B.분석적인 C.동정심이 많은 D.단호한
# 16. A.지도력 있는 B.충동적인 C.느린 D.비판적인
# 17. A.일관성 있는 B.영향력있는 C.생기있는 D.느긋한
# 18. A.유력한 B.친절한 C.독립적인 D.정돈된
# 19. A.이상주의적인 B.평판이 좋은 C.쾌활한 D.솔직한
# 20. A.참을성 없는 B.진지한 C.미루는 D.감성적인
# 21. A.경쟁심이 있는 B.자발적인 C.충성스러운 D.사려깊은
# 22. A.희생적인 B.이해심 많은 C.설득력 있는 D.용기있는
# 23. A.의존적인 B.변덕스러운 C.절제력 있는 D.밀어붙이는
# 24. A.포용력 있는 B.전통적인 C.사람을 부추기는 D.이끌어 가는

# -----------------------------------------------------------------------------
# 함수기능 : 검사지 메인화면 함수
# 함수이름 : print_start
# 매개변수 : 없음 
# 함수결과 : 메인화면 출력
# -----------------------------------------------------------------------------

def print_start():
    print(f'{"-":-^26}')
    print(f'{"|     D I S C 검 사      |":^20}')
    print(f'{"-":-^26}')
    print(f'{"| Dominant      : 주도형 |":^20}')
    print(f'{"| Influential   : 사교형 |":^20}')
    print(f'{"| Steady        : 안정형 |":^20}')
    print(f'{"| Conscientious : 신중형 |":^20}')
    print(f'{"-":-^26}')

# -----------------------------------------------------------------------------
# 함수기능 : 입력 받은 데이터가 유효한 데이터인지 검사하는 함수
# 함수이름 : check_data
# 매개변수 : str 데이터 체크(data), 데이터 수(count) 
# 함수결과 : True, False
# -----------------------------------------------------------------------------

check_list=['A','B','C','D']
def check_data(data, count):
    if len(data) == count:
        if data in check_list:
            return True
        else:
            print('"A", "B", "C", "D" 중에 하나만 입력하세요.')
            return False
    else:
        print('"A", "B", "C", "D" 중에 하나만 입력하세요.')
        return False

# -----------------------------------------------------------------------------
# 함수기능 : D I S C 값 계산 함수
# 함수이름 : calc_data
# 매개변수 : number, v1, v2, v3, v4
# 함수결과 : disc 각 항목 점수 합산
# -----------------------------------------------------------------------------

disc=[0,0,0,0]
def calc_data(number, v1, v2, v3, v4):
    if answer_list[number] == 'A': disc[v1]+=1
    elif answer_list[number] == 'B': disc[v2]+=1
    elif answer_list[number] == 'C': disc[v3]+=1
    else: disc[v4]+=1
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 최대값 키 가져오기
# -----------------------------------------------------------------------------
# largest_number_key=""
# largest_number=max(score_dict.values()) # 최대값 가져오기

# if max(score_dict.values()) == domi: largest_number_key="D"
# elif max(score_dict.values()) == infl: largest_number_key="I"
# elif max(score_dict.values()) == stea: largest_number_key="S"
# elif max(score_dict.values()) == cons: largest_number_key="C"

# -----------------------------------------------------------------------------
# 함수기능 : 최대값 키 가져오기 2 (리스트 컴프리헨션)
# 함수이름 : largest_number
# 매개변수 : domi, infl, stea, cons
# 함수결과 : 최대값의 키를 모두 가져옴 (최대값이 1개일 때, 2개일 때 모두 가능)
# -----------------------------------------------------------------------------

def largest_number(domi, infl, stea, cons):
    score_dict={'D':domi, "I":infl, "S":stea, "C":cons}
    largest_number_key=[key for key, value in score_dict.items() if max(score_dict.values()) == value]
    return largest_number_key

# -----------------------------------------------------------------------------
# 함수기능 : 두번째로 큰 값 키 가져오기
# 함수이름 : second_large_number
# 매개변수 : domi, infl, stea, cons
# 함수결과 : 두번째로 큰 값의 키를 가져옴
# -----------------------------------------------------------------------------

def second_large_number4(domi, infl, stea, cons):
    if largest_number(domi, infl, stea, cons)[0] == 'D':
        if domi-infl == domi-stea:
            return 'I','S'
        elif domi-infl == domi-cons:
            return 'I','C'
        elif domi-cons== domi-stea:
            return 'C','S'
        else:
            if min(domi-infl, domi-stea, domi-cons) == domi-infl:
                return 'I'
            elif min(domi-infl, domi-stea, domi-cons) == domi-stea:
                return 'S'
            else:
                return 'C'
    elif largest_number(domi, infl, stea, cons)[0] == 'I':
        if infl-domi == infl-stea:
            return 'D','S'
        elif infl-domi == infl-cons:
            return 'D','C'
        elif infl-cons == infl-stea:
            return 'C','S'
        else:
            if min(infl-domi, infl-stea, infl-cons) == infl-domi:
                return 'D'
            elif min(infl-domi, infl-stea, infl-cons) == infl-stea:
                return 'S'
            else:
                return 'C'
    elif largest_number(domi, infl, stea, cons)[0] == 'S':
        if stea-domi == stea-infl:
            return 'D','I'
        elif stea-infl == stea-cons:
            return 'I','C'
        elif stea-domi == stea-cons:
            return 'D','C'
        else:
            if min(stea-domi, stea-infl, stea-cons) == stea-domi:
                return 'D'
            elif min(stea-domi, stea-infl, stea-cons) == stea-infl:
                return 'I'
            else:
                return 'C'
    else: # largest_number(domi, infl, stea, cons)[0] == cons
        if cons-domi == cons-infl:
            return 'D','I'
        elif cons-domi == cons-stea:
            return 'D','S'
        elif cons-infl == cons-stea:
            return 'I','S'
        else:
            if min(cons-domi, cons-infl, cons-stea) == cons-domi:
                return 'D'
            elif min(cons-domi, cons-infl, cons-stea) == cons-infl:
                return 'I'
            else:
                return 'S'

# def second_large_number3(domi, infl, stea, cons):
    if largest_number(domi, infl, stea, cons)[0] == 'D':
        # domi-infl, domi-stea, domi-cons => 이중 가장 작은 숫자가 두번째로 큰 값
        if min(domi-infl, domi-stea, domi-cons) == domi-infl:
            return 'I'
        elif min(domi-infl, domi-stea, domi-cons) == domi-stea:
            return 'S'
        else:
            return 'C'
    elif largest_number(domi, infl, stea, cons)[0] == 'I':
        if min(infl-domi, infl-stea, infl-cons) == infl-domi:
            return 'D'
        elif min(infl-domi, infl-stea, infl-cons) == infl-stea:
            return 'S'
        else:
            return 'C'
    elif largest_number(domi, infl, stea, cons)[0] == 'S':
        if min(stea-domi, stea-infl, stea-cons) == stea-domi:
            return 'D'
        elif min(stea-domi, stea-infl, stea-cons) == stea-infl:
            return 'I'
        else:
            return 'C'
    else: # largest_number(domi, infl, stea, cons)[0] == cons
        if min(cons-domi, cons-infl, cons-stea) == cons-domi:
            return 'D'
        elif min(cons-domi, cons-infl, cons-stea) == cons-infl:
            return 'I'
        else:
            return 'S'


# def second_large_number2(domi, infl, stea, cons):
    if largest_number(domi, infl, stea, cons)[0] == domi:
        if infl == stea: return ['I','S']
        elif infl == cons: return ['I','C']
        else: return ['S','C']
    elif largest_number(domi, infl, stea, cons)[0] == infl:
        if domi == stea: return ['D','S']
        elif domi == cons: return ['D','C']
        else: return ['S','C']
    elif largest_number(domi, infl, stea, cons)[0] == stea:
        if domi == infl: return ['D','I']
        elif domi == cons: return ['D','C']
        else: return ['I','C']
    else: 
        if domi == infl: return ['D','I']
        elif infl == stea: return ['I','S']
        else: return ['D','S']


# def second_large_number(domi, infl, stea, cons):
    if largest_number(domi, infl, stea, cons)[0] == domi:
        if max(infl, stea, cons) == infl: return 'I'
        elif max(infl, stea, cons) == stea: return 'S'
        elif max(infl, stea, cons) == cons: return 'C'
    elif largest_number(domi, infl, stea, cons)[0] == infl:
        if max(domi, stea, cons) == domi: return 'D'
        elif max(domi, stea, cons) == stea: return 'S'
        elif max(domi, stea, cons) == cons: return 'C'
    elif largest_number(domi, infl, stea, cons)[0] == stea:
        if max(domi, infl, cons) == domi: return 'D'
        elif max(domi, infl, cons) == infl: return 'I'
        elif max(domi, infl, cons) == cons: return 'C'
    else: 
        if max(domi, infl, stea) == domi: return 'D'
        elif max(domi, infl, stea) == infl: return 'I'
        elif max(domi, infl, stea) == stea: return 'S'


# -----------------------------------------------------------------------------
### 질문 초기 컨셉
# -----------------------------------------------------------------------------
# def ask_first():
    # ask1=input('1. A.절제하는 B.강력한 C.꼼꼼한 D.표현력있는 : ')
    # ask2=input('2. A.개척적인 B.정확한 C.흥미진진한 D.만족스러운 : ')
    # ask3=input('3. A.기꺼이하는 B.활기있는 C.대담한 D.정교한 : ')
    # ask4=input('4. A.논쟁을 좋아하는 B.회의적인 C.주저하는 D.예측할 수 없는 : ')
    # ask5=input('5. A.공손한 B.사교적인 C.참을성이 있는 D.무서움을 모르는 : ')
    # ask6=input('6. A.설득력 있는 B.독립심이 강한 C.논리적인 D.온화한 : ')
    # ask7=input('7. A.신중한 B.차분한 C.과단성 있는 D.파티를 좋아하는 : ')
    # ask8=input('8. A.인기있는 B.고집있는 C.완벽주의자 D.인심 좋은 : ')
    # ask9=input('9. A.변화가 많은 B.수줍음을 타는 C.느긋한 D.완고한 : ')
    # ask10=input('10. A.체계적인 B.낙관적인 C.의지가 강한 D.친절한 : ')
    # ask11=input('11. A.엄격한 겸손한 상냥한 말주변이 좋은 : ')
    # ask12=input('12. A.호의적인 B.빈틈없는 C.놀기 좋아하는 D.의지가 강한 : ')
    # ask13=input('13. A.참신한 B.모험적인 C.절제된 D.신중한 : ')
    # ask14=input('14. A.참는 B.성실한 C.공격적인 D.매력있는 : ')
    # ask15=input('15. A.열정적인 B.분석적인 C.동정심이 많은 D.단호한 : ')
    # ask16=input('16. A.지도력 있는 B.충동적인 C.느린 D.비판적인 : ')
    # ask17=input('17. A.일관성 있는 B.영향력있는 C.생기있는 D.느긋한 : ')
    # ask18=input('18. A.유력한 B.친절한 C.독립적인 D.정돈된 : ')
    # ask19=input('19. A.이상주의적인 B.평판이 좋은 C.쾌활한 D.솔직한 : ')
    # ask20=input('20. A.참을성 없는 B.진지한 C.미루는 D.감성적인 : ')
    # ask21=input('21. A.경쟁심이 있는 B.자발적인 C.충성스러운 D.사려깊은 : ')
    # ask22=input('22. A.희생적인 B.이해심 많은 C.설득력 있는 D.용기있는 : ')
    # ask23=input('23. A.의존적인 B.변덕스러운 C.절제력 있는 D.밀어붙이는 : ')
    # ask24=input('24. A.포용력 있는 B.전통적인 C.사람을 부추기는 D.이끌어 가는 : ')
# -----------------------------------------------------------------------------
### 계산 초기 컨셉
# -----------------------------------------------------------------------------
# def calc_first():
    # domi = 0 # 주도형 기본값
    # infl = 0 # 사교형 기본값
    # stea = 0 # 안정형 기본값
    # cons = 0 # 신중형 기본값

    # if answer_list[0] == 'A': stea += 1
    # elif answer_list[0] == 'B': domi += 1
    # elif answer_list[0] == 'C': cons += 1
    # else: infl += 1

    # if answer_list[1] == 'A': domi += 1
    # elif answer_list[1] == 'B': cons += 1
    # elif answer_list[1] == 'C': infl += 1
    # else: stea += 1

    # if answer_list[2] == 'A': stea += 1
    # elif answer_list[2] == 'B': infl += 1
    # elif answer_list[2] == 'C': domi += 1
    # else: cons += 1

    # if answer_list[3] == 'A': domi += 1
    # elif answer_list[3] == 'B': cons += 1
    # elif answer_list[3] == 'C': stea += 1
    # else: infl += 1

    # if answer_list[4] == 'A': cons += 1
    # elif answer_list[4] == 'B': infl += 1
    # elif answer_list[4] == 'C': stea += 1
    # else: domi += 1

    # if answer_list[5] == 'A': infl += 1
    # elif answer_list[5] == 'B': domi += 1
    # elif answer_list[5] == 'C': cons += 1
    # else: stea += 1

    # if answer_list[6] == 'A': cons += 1
    # elif answer_list[6] == 'B': stea += 1
    # elif answer_list[6] == 'C': domi += 1
    # else: infl += 1

    # if answer_list[7] == 'A': infl += 1
    # elif answer_list[7] == 'B': domi += 1
    # elif answer_list[7] == 'C': cons += 1
    # else: stea += 1

    # if answer_list[8] == 'A': infl += 1
    # elif answer_list[8] == 'B': cons += 1
    # elif answer_list[8] == 'C': stea += 1
    # else: domi += 1

    # if answer_list[9] == 'A': cons += 1
    # elif answer_list[9] == 'B': infl += 1
    # elif answer_list[9] == 'C': domi += 1
    # else: stea += 1

    # if answer_list[10] == 'A': domi += 1
    # elif answer_list[10] == 'B': cons += 1
    # elif answer_list[10] == 'C': stea += 1
    # else: infl += 1

    # if answer_list[11] == 'A': stea += 1
    # elif answer_list[11] == 'B': cons += 1
    # elif answer_list[11] == 'C': infl += 1
    # else: domi += 1

    # if answer_list[12] == 'A': infl += 1
    # elif answer_list[12] == 'B': domi += 1
    # elif answer_list[12] == 'C': cons += 1
    # else: stea += 1

    # if answer_list[13] == 'A': cons += 1
    # elif answer_list[13] == 'B': stea += 1
    # elif answer_list[13] == 'C': domi += 1
    # else: infl += 1

    # if answer_list[14] == 'A': infl += 1
    # elif answer_list[14] == 'B': cons += 1
    # elif answer_list[14] == 'C': stea += 1
    # else: domi += 1

    # if answer_list[15] == 'A': domi += 1
    # elif answer_list[15] == 'B': infl += 1
    # elif answer_list[15] == 'C': stea += 1
    # else: cons += 1

    # if answer_list[16] == 'A': cons += 1
    # elif answer_list[16] == 'B': domi += 1
    # elif answer_list[16] == 'C': infl += 1
    # else: stea += 1

    # if answer_list[17] == 'A': infl += 1
    # elif answer_list[17] == 'B': stea += 1
    # elif answer_list[17] == 'C': domi += 1
    # else: cons += 1

    # if answer_list[18] == 'A': cons += 1
    # elif answer_list[18] == 'B': infl += 1
    # elif answer_list[18] == 'C': stea += 1
    # else: domi += 1

    # if answer_list[19] == 'A': domi += 1
    # elif answer_list[19] == 'B': cons += 1
    # elif answer_list[19] == 'C': stea += 1
    # else: infl += 1

    # if answer_list[20] == 'A': domi += 1
    # elif answer_list[20] == 'B': infl += 1
    # elif answer_list[20] == 'C': stea += 1
    # else: cons += 1

    # if answer_list[21] == 'A': cons += 1
    # elif answer_list[21] == 'B': stea += 1
    # elif answer_list[21] == 'C': infl += 1
    # else: domi += 1

    # if answer_list[22] == 'A': cons += 1
    # elif answer_list[22] == 'B': infl += 1
    # elif answer_list[22] == 'C': domi += 1
    # else: stea += 1

    # if answer_list[23] == 'A': cons += 1
    # elif answer_list[23] == 'B': stea += 1
    # elif answer_list[23] == 'C': domi += 1
    # else: infl += 1
# -----------------------------------------------------------------------------
# 질문지 리스트 정리 
# 24개 질문변수를 2차원 리스트로
# -----------------------------------------------------------------------------

ask1=['A.절제하는', 'B.강력한', 'C.꼼꼼한', 'D.표현력있는']
ask2=['A.개척적인', 'B.정확한', 'C.흥미진진한', 'D.만족스러운']
ask3=['A.기꺼이하는', 'B.활기있는', 'C.대담한', 'D.정교한']
ask4=['A.논쟁을 좋아하는', 'B.회의적인', 'C.주저하는', 'D.예측할 수 없는']
ask5=['A.공손한', 'B.사교적인', 'C.참을성이 있는', 'D.무서움을 모르는']
ask6=['A.설득력 있는', 'B.독립심이 강한', 'C.논리적인', 'D.온화한']
ask7=['A.신중한', 'B.차분한', 'C.과단성 있는', 'D.파티를 좋아하는']
ask8=['A.인기있는', 'B.고집있는', 'C.완벽주의자', 'D.인심 좋은']
ask9=['A.변화가 많은', 'B.수줍음을 타는', 'C.느긋한', 'D.완고한']
ask10=['A.체계적인', 'B.낙관적인', 'C.의지가 강한', 'D.친절한']
ask11=['A.엄격한', 'B.겸손한', 'C.상냥한', 'D.말주변이 좋은']
ask12=['A.호의적인', 'B.빈틈없는', 'C.놀기 좋아하는', 'D.의지가 강한']
ask13=['A.참신한', 'B.모험적인', 'C.절제된', 'D.신중한']
ask14=['A.참는', 'B.성실한', 'C.공격적인', 'D.매력있는']
ask15=['A.열정적인', 'B.분석적인', 'C.동정심이 많은', 'D.단호한']
ask16=['A.지도력 있는', 'B.충동적인', 'C.느린', 'D.비판적인']
ask17=['A.일관성 있는', 'B.영향력있는', 'C.생기있는', 'D.느긋한']
ask18=['A.유력한', 'B.친절한', 'C.독립적인', 'D.정돈된']
ask19=['A.이상주의적인', 'B.평판이 좋은', 'C.쾌활한', 'D.솔직한']
ask20=['A.참을성 없는', 'B.진지한', 'C.미루는', 'D.감성적인']
ask21=['A.경쟁심이 있는', 'B.자발적인', 'C.충성스러운', 'D.사려깊은']
ask22=['A.희생적인', 'B.이해심 많은', 'C.설득력 있는', 'D.용기있는']
ask23=['A.의존적인', 'B.변덕스러운', 'C.절제력 있는', 'D.밀어붙이는']
ask24=['A.포용력 있는', 'B.전통적인', 'C.사람을 부추기는', 'D.이끌어 가는']

asks=[ask1,ask2,ask3,ask4,ask5,ask6,ask7,ask8,ask9,ask10,ask11,ask12,ask13,ask14,\
      ask15,ask16,ask17,ask18,ask19,ask20,ask21,ask22,ask23,ask24]

# A_list=[]
# B_list=[]
# C_list=[]
# D_list=[]

# for i in range(len(asks)):
#     A_list.append(asks[i][0])
#     B_list.append(asks[i][1])
#     C_list.append(asks[i][2])
#     D_list.append(asks[i][3])

# -----------------------------------------------------------------------------
# 프로그램 실행
# -----------------------------------------------------------------------------

print_start() # 메인화면
print('차례로 자신을 가장 잘 표현한다고 생각하는 단어를 선택합니다. (ex. A.절제하는 B.강력한 C.꼼꼼한 D.표현력있는) : A')
i=0
answer_list=[]
while i < len(asks):
    asks[i]=input(f'{i+1}. {asks[i]} : ')
    answer_list.append(asks[i])
    if check_data(asks[i], 1) == True:
        i += 1
        continue

calc_data(0, 2, 0, 3, 1)
calc_data(1, 0, 3, 1, 2)
calc_data(2, 2, 1, 0, 3)
calc_data(3, 0, 3, 2, 1)
calc_data(4, 3, 1, 2, 0)
calc_data(5, 1, 0, 3, 2)
calc_data(6, 3, 2, 0, 1)
calc_data(7, 1, 0, 3, 2)
calc_data(8, 1, 3, 2, 0)
calc_data(9, 3, 1, 0, 2)
calc_data(10, 0, 3, 2, 1)
calc_data(11, 2, 3, 1, 0)
calc_data(12, 1, 0, 3, 2)
calc_data(13, 3, 2, 0, 1)
calc_data(14, 1, 3, 2, 0)
calc_data(15, 0, 1, 2, 3)
calc_data(16, 3, 1, 0, 2)
calc_data(17, 1, 2, 0, 3)
calc_data(18, 3, 1, 2, 0)
calc_data(19, 0, 3, 2, 1)
calc_data(20, 0, 1, 2, 3)
calc_data(21, 3, 2, 1, 0)
calc_data(22, 3, 1, 0, 2)
calc_data(23, 3, 2, 0, 1)

# -----------------------------------------------------------------------------

print()
print_start()
print()
print(f'주도형(D) 점수는 {disc[0]}입니다.')
print(f'사교형(I) 점수는 {disc[1]}입니다.')
print(f'안정형(S) 점수는 {disc[2]}입니다.')
print(f'신중형(C) 점수는 {disc[3]}입니다.')
print()


if len(largest_number(disc[0], disc[1], disc[2], disc[3])) == 1:
    print(f'당신의 가장 높은 점수를 차지한 스타일은 {largest_number(disc[0], disc[1], disc[2], disc[3])[0]}입니다.')
    print(f'당신의 두번째 높은 점수를 차지한 스타일은 {second_large_number4(disc[0], disc[1], disc[2], disc[3])}입니다.')
elif len(largest_number(disc[0], disc[1], disc[2], disc[3])) == 2:
    print(f'당신의 가장 높은 점수를 차지한 스타일은 {largest_number(disc[0], disc[1], disc[2], disc[3])[0]}와 {largest_number(disc[0], disc[1], disc[2], disc[3])[1]}입니다.')
else:
    print(f'당신의 가장 높은 점수를 차지한 스타일은 {largest_number(disc[0], disc[1], disc[2], disc[3])[0]}와 {largest_number(disc[0], disc[1], disc[2], disc[3])[1]}와 {largest_number(disc[0], disc[1], disc[2], disc[3])[2]}입니다.')