'''
- 1벌의 카드(deck) 생성: 리스트로 구현
- 각 Player들에게 카드를 나누어 주는 기능
    - 자신이 가진 deck에서 제거하여 다른 선수들에게 제공
'''

from card import Card
import random

class GameDealer:

    def __init__(self):
        self.deck = list()
        self.suit_number = 13
        
    # def __repr__(self):
    #     '''
    #     객체를 공식적인 문자열로 변환(인터프리터에서 객체 표현에 사용)
    #     '''
    #     return f'{self.deck}'

    def make_deck(self):
        card_suits = ['♠', '♥', '♣', '◆']
        card_numbers = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        for i in range(len(card_suits)):
            for j in range(self.suit_number):
                self.deck.append(Card(card_suits[i], card_numbers[j]))
        print(f'[GameDealer] 초기 카드 생성')
        self.print_deck()

    # def shuffle_card(self): 함수 하나에 다 넣기
        print(f'[GameDealer] 카드 랜덤하게 섞기')
        random.shuffle(self.deck)
        self.print_deck()

    def distribute_card(self, num):
        print()
        print('=' * 60)
        print(f'카드 나누어 주기: {num}장')

        card_list1 = list()
        card_list2 = list()

        for i in range(num):
            card_list1.append(self.deck.pop())
            card_list2.append(self.deck.pop())

        self.print_deck()
        return card_list1, card_list2

    def print_deck(self):
        print('-' * 60)
        print(f'[GameDealer] 딜러가 가진 카드 수: {len(self.deck)}')
        for i in range(len(self.deck)):
            print(self.deck[i], end=' ')
            if (i + 1) % self.suit_number == 0:
                print()
        print()

# if __name__ == '__main__':
#     cardes = GameDealer()
#     cardes.make_deck()
#     print('[Game Dealer] 초기 카드 생성')
#     print('-'*100)
#     print('[Game Dealer] 딜러가 가진 카드 수: 52\n')
#     print(cardes)
#     print('[Game Dealer] 카드 랜덤하게 섞기')
#     print('-'*100)
#     print('[Game Dealer] 딜러가 가진 카드 수: 52\n')
#     cardes.shuffle_card()
#     print(cardes)