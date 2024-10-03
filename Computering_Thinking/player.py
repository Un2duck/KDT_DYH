'''
- 자신이 가지고 있는 카드 관리
    - 두 개의 리스트를 가짐 (holding_card_list,	open_card_list)
- 두 장의 동일한 카드를 제거하는 기능
    > 번호가 동일한 경우, holding_card_list에서 open_care_list로 이동: 테이블에 공개하는 기능
- 두 개의 리스트를 출력하는 기능
'''

from gamedealer import GameDealer


class Player:
    def __init__(self, name):
        self.name = name
        self.holding_card_list = list()
        self.open_card_list = list()

    def add_card_list(self, card_list):
        for card in card_list:
            self.holding_card_list.append(card)
    
    def display_two_card_lists(self):
        print('=' * 60)

        print(f'[{self.name}] Open card list: {len(self.open_card_list)}')
        for card in self.open_card_list:
            print(card, end=' ')
        print()
        print()

        print(f'[{self.name}] Holding card list: {len(self.holding_card_list)}')
        for card in self.holding_card_list:
            print(card, end=' ')
        print()
        print()

        # back_list = [self.holding_card_list[i].number for i in range(10)]
        # same = [i for i in set(back_list) if back_list.count(i) == 2]
        # x = [self.holding_card_list[i] for i in range(10) if self.holding_card_list[i].number in same]
        
        # # open_card_list로 이동.
        # self.open_card_list.extend(x)
        # print(f'[{name}] Open card list: \n', self.open_card_list)
        
        # # open_card_list의 카드들 반환.
        # return x
    
    def check_one_pair_card(self):
        print('=' * 60)
        print(f'[{self.name}: 숫자가 같은 한쌍의 카드 검사]')
        for i in range(0, len(self.holding_card_list) - 1):
            card1 = self.holding_card_list[i]
            for j in range(i + 1, len(self.holding_card_list)):
                card2 = self.holding_card_list[j]
                if card1.number == card2.number:
                    if card1 not in self.open_card_list and card2 not in self.open_card_list:
                        self.open_card_list.append(card1)
                        self.open_card_list.append(card2)
        
        # open_card_list에 저장한 항목들을 holding_card_list에서 제거
        for card in self.open_card_list:
            if card in self.holding_card_list:
                self.holding_card_list.remove(card)
        
        self.display_two_card_lists()
    
    def get_holding_card_count(self):
        return len(self.holding_card_list)
    
    def get_open_card_count(self):
        return len(self.open_card_list)



# # 딜러 선언
# dealer = GameDealer()
# dealer.make_deck()

# print('[Game Dealer] 초기 카드 생성')
# print('-'*100)
# print('[Game Dealer] 딜러가 가진 카드 수: 52\n')
# print(dealer)
# print('[Game Dealer] 카드 랜덤하게 섞기')
# print('-'*100)
# print('[Game Dealer] 딜러가 가진 카드 수: 52\n')

# # 카드 섞기
# dealer.shuffle_card()
# print(dealer)

# # 플레이어들 선언
# player1 = Player('흥부')
# player2 = Player('놀부')

# player1.add_card_list(dealer.deck)
# player2.add_card_list(dealer.deck)

# print('[흥부] Holding card list:\n',player1.holding_card_list)

# print('[놀부] Holding card list:\n',player2.holding_card_list)

# print('-'*100)

# print('[2단계]')

# player1.display_two_card_lists('흥부')

# player2.display_two_card_lists('놀부')

# # holding card list 안에서 open card list에 있는 요소들 빼기
# changed = [card for card in player1.holding_card_list if card not in player1.display_two_card_lists('흥부')]

# # holding card list 갱신
# player1.holding_card_list = changed

# print('[흥부] Holding card list:\n',player1.holding_card_list)