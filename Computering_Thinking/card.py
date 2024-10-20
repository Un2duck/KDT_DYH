'''
- 한 장의 카드를 나타내기 위한 클래스
- suit와 number의 값 가짐
'''

class Card:
    def __init__(self, card_suit, card_number):
        self.suit = card_suit
        self.number = card_number

    def __repr__(self):
        '''
        객체를 공식적인 문자열로 변환(인터프리터에서 객체 표현에 사용)
        '''
        return f'({self.suit},{self.number:>2})'
    
if __name__ == '__main__':
    card = Card('♠', 2)
    print(card)