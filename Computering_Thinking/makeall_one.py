import random

player1 = '흥부'
player2 = '놀부'
list1 = list()
list2 = list()
display1 = list()
display2 = list()

card_list = []
def make_deck():
    card_suits = ['♠', '♥', '♣', '◆']
    card_numbers = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

    for i in range(len(card_suits)):
        for j in range(len(card_numbers)):
            card_list.append((card_suits[i], card_numbers[j]))
    return card_list

def shuffle_deck():
    random.shuffle(card_list)

def prepare_card():
    for i in range(20): list1.append(card_list.pop(0)) if i % 2 == 0 else list2.append(card_list.pop(0))

make_deck()
shuffle_deck()
print(card_list)

print('='*100)

prepare_card()
print('list1:',list1)
# print('list2:',list2)
# print(card_list)
# print('현재 카드더미 수:', len(card_list))

# 같은 숫자끼리 묶어내기
# 숫자만 다 가져오기
back_list1 = [list1[i][1] for i in range(10)]

# 2개 이상 중복 식 만들기
same1 = [i for i in set(back_list1) if back_list1.count(i) >= 2]

b = [list1[i] for i in range(10) if list1[i][1] in same1]
display1.extend(b)
print('display1:',display1)

filtered_list1 = [card for card in list1 if card not in display1]
print(filtered_list1)