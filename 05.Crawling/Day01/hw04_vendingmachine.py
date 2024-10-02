'''
1. 클래스를 이용한 커피 자판기 프로그램 설계 및 구현
제출 파일명: hw04_vendingmachine.py
모든 커피의 가격은 300원으로 동일하며 선택한 메뉴에 따라 각 물품의 소모량은 아래와 같이 소모가 됩니다.

충분한 돈을 입력하지 않거나, 
자판기 내부의 물품(커피, 프림, 설탕, 물, 종이컵 등)이 부족한 경우에는 재료가 부족하다는 메시지를 출력하고
프로그램을 종료 한다.
사용자가 선택한 커피 메뉴에 따라 각 물품의 소모량은 다르며, 
커피를 제공하면 선택 메뉴에 따른 재료 현황(딕셔너리 형태)을 업데이트하며, 종료 메뉴를 선택하면 남은 돈을 반환한다.
아래 각 기능들은 클래스의 메소드로 구현하고, 필요한 메소드는 추가 구현하면 됨

'''

'''
초기 자판기 재료 현황: 아래 물품은 모두 딕셔너리로 구현
- 자판기 내부 동전: 0원 (사용자가 커피를 먹을 때 마다 300원씩 증가)
- 물: 500ml
- 커피: 100g, 프림: 100g, 설탕: 100g, 종이컵 5개
- 딕셔너리는 클래스 생성자에 선언 후 각 함수들에서 사용
'''

'''
메뉴 출력 및 선택 기능 (종료를 선택할 때까지 반복)
    - 초기에 1회만 동전을 투입하고 메뉴를 선택
    - 투입된 돈이 300원 이상인 경우에만 커피 제공
    - 메뉴 출력시 현재 잔액을 화면에 표시
    - 메뉴 1. 블랙 커피, 2: 프림 커피, 3: 설탕 프림 커피, 4: 재료 현황, 5: 종료
커피 제공 기능: 메뉴에 따른 커피, 설탕, 프림 소모량
    - 먼저 자판기에 남은 재료의 양을 검사한 다음, 선택한 메뉴에 따라 충분한
    재료가 남아 있는 경우에 한해서 커피를 제공하며 커피를 제공한 다음 재료
    현황을 업데이트하고 화면에 출력
    - 블랙 커피: 커피 30g + 물 100ml
    - 프림 커피: 커피 15g + 프림 15g + 물 100ml
    - 설탕 프림 커피: 커피 10g + 프림 10g + 설탕 10g + 물 100ml
재료 현황 업데이트 기능: 딕셔너리 업데이트
    - 커피, 프림, 설탕, 컵, 잔여 물 용량 업데이트
재료 현황 출력 기능
    - 커피를 제공하면 현재 자판기에 남아 있는 커피량, 프림량, 설탕량, 컵의 개수,
    남은 물 용량 출력
물품 현황 체크 기능
    - 사용자가 선택한 커피 메뉴에 필요한 물품 현황 체크
    - 충분한 물품이 없는 경우, “재료가 부족합니다.” 출력하고 남은 돈을 반환한
    다음 프로그램을 종료
'''

class VendingMachine:
    def __init__(self, input_dict):
        '''
        생성자
        :param input_dict: 초기 자판기 재료량(dict 형태)
        '''
        self.input_money = 0
        self.inventory = input_dict

    def	run(self):
        """
        커피 자판기 동작 및 메뉴 호출 함수
        """
        coin = int(input('동전을 투입하세요.: '))
        while True:
            self.menu(coin)
            # 300원보다 부족할 경우 종료
            if coin < 300:
                print(f'투입된 돈 ({coin})원이 300원보다 작습니다.')
                print(f'{coin}원을 반환합니다.')
                print('-'*50)
                print('커피 자판기 동작을 종료합니다.')
                print('-'*50)
                break

            elif inventory_dict['coffee'] < 0 or inventory_dict['cream'] < 0 or inventory_dict['sugar'] < 0 or \
                inventory_dict['water'] < 0 or inventory_dict['cup'] < 0:
                print('재료가 부족합니다.')
                print('-'*50)
                self.show_stock()
                print('-'*50)
                print(f'{coin}원을 반환합니다.')
                print('-'*50)
                print('커피 자판기 동작을 종료합니다.')
                print('-'*50)
                break

            else:
                choice = input('메뉴를 선택하세요.: ')
                if choice == '5':
                    print(f'종료를 선택했습니다. {coin}원이 반환됩니다.')
                    print('커피 자판기 동작을 종료합니다.')
                    break

                elif choice == '1':
                    coin = coin-300 # 커피값 300원 차감
                    inventory_dict['change'] += 300
                    print(f'블랙 커피를 선택하셨습니다. 잔액: {coin}')
                    # 사용하는 재고) 커피 -30, 물 -100, 컵 -1
                    self.stock(inven=inventory_dict, coffee='using1', water='using4', 
                        cup='using5', amount1=30, amount4=100, amount5=1)
                    self.show_stock()

                elif choice == '2':
                    coin = coin-300 # 커피값 300원 차감
                    inventory_dict['change'] += 300
                    print(f'프림 커피를 선택하셨습니다. 잔액: {coin}')
                    # 사용하는 재고) 커피 -15, 프림 -15, 물 -100, 컵 -1
                    self.stock(inven=inventory_dict, coffee='using1', cream='using2', water='using4', 
                        cup='using5', amount1=15, amount2=15, amount4=100, amount5=1)
                    self.show_stock()

                elif choice == '3':
                    coin = coin-300 # 커피값 300원 차감
                    inventory_dict['change'] += 300
                    print(f'설탕 프림 커피를 선택하셨습니다. 잔액: {coin}')
                    # 사용하는 재고) 커피 -10, 프림 -10, 설탕 -10, 물 -100, 컵 -1
                    self.stock(inven=inventory_dict, coffee='using1', cream='using2', sugar='using3', water='using4', 
                        cup='using5', amount1=15, amount2=15, amount3=10, amount4=100, amount5=1)
                    self.show_stock()

                elif choice == '4':
                    self.show_stock()

    # 기능 구현 및 다른 메소드 호출

    # 메뉴 함수
    def menu(self, coin):
        print('-'*50)
        print(f'커피 자판기 (잔액: {coin}원)')
        print('-'*50)
        print('1. 블랙 커피')
        print('2. 프림 커피')
        print('3. 설탕 프림 커피')
        print('4. 재료 현황')
        print('5. 종료')

    # 재고 조정 함수 (1:커피, 2:프림, 3:설탕, 4:물, 5:컵)
    def stock(self, **kwargs):
        result = kwargs.get('inven', {})
        for key, oper in kwargs.items():
            if key != 'inven' and key in result:
                if oper == 'using1':
                    result[key] -= kwargs.get('amount1', 0)
                if oper == 'using2':
                    result[key] -= kwargs.get('amount2', 0)
                if oper == 'using3':
                    result[key] -= kwargs.get('amount3', 0)
                if oper == 'using4':
                    result[key] -= kwargs.get('amount4', 0)
                if oper == 'using5':
                    result[key] -= kwargs.get('amount5', 0)

    # 재료 현황 함수
    def show_stock(self):
        print('재료 현황:', end=' ')
        for key, value in inventory_dict.items():
            print(f'{key}: {value}', end='  ')
        print()

# VendingMachine 객체 생성
inventory_dict = {'coffee':	100, 'cream': 100, 'sugar': 100,
                  'water':	500, 'cup':	5, 'change': 0}
coffee_machine = VendingMachine(inventory_dict)
coffee_machine.run() # VendingMachine 동작 메소드


# -----------------------------------------------------------------------------
# 함수 구현
# -----------------------------------------------------------------------------

# 메뉴 함수
def menu(coin):
        print('-'*50)
        print(f'커피 자판기 (잔액: {coin}원)')
        print('-'*50)
        print('1. 블랙 커피')
        print('2. 프림 커피')
        print('3. 설탕 프림 커피')
        print('4. 재료 현황')
        print('5. 종료')

# 재고 조정 함수 (1:커피, 2:프림, 3:설탕, 4:물, 5:컵)
def stock(**kwargs):
    result = kwargs.get('inven', {})
    for key, oper in kwargs.items():
        if key != 'inven' and key in result:
            if oper == 'using1':
                result[key] -= kwargs.get('amount1', 0)
            if oper == 'using2':
                result[key] -= kwargs.get('amount2', 0)
            if oper == 'using3':
                result[key] -= kwargs.get('amount3', 0)
            if oper == 'using4':
                result[key] -= kwargs.get('amount4', 0)
            if oper == 'using5':
                result[key] -= kwargs.get('amount5', 0)

# 재료 현황 함수
def show_stock():
    print('재료 현황:', end=' ')
    for key, value in inventory_dict.items():
        print(f'{key}: {value}', end='  ')
    print()
#
# -----------------------------------------------------------------------------
# coin = int(input('동전을 투입하세요.: '))
# while True:
#     menu(coin)
#     # 300원보다 부족할 경우 종료
#     if coin < 300:
#         print(f'투입된 돈 ({coin})원이 300원보다 작습니다.')
#         print(f'{coin}원을 반환합니다.')
#         print('-'*50)
#         print('커피 자판기 동작을 종료합니다.')
#         print('-'*50)
#         break
#     elif inventory_dict['coffee'] < 0 or inventory_dict['cream'] < 0 or inventory_dict['sugar'] < 0 or \
#         inventory_dict['water'] < 0 or inventory_dict['cup'] < 0:
#         print('재료가 부족합니다.')
#         print('-'*50)
#         show_stock()
#         print('-'*50)
#         print(f'{coin}원을 반환합니다.')
#         print('-'*50)
#         print('커피 자판기 동작을 종료합니다.')
#         print('-'*50)
#         break
#     else:
#         choice = input('메뉴를 선택하세요.: ')
#         if choice == '5':
#             print(f'종료를 선택했습니다. {coin}원이 반환됩니다.')
#             print('커피 자판기 동작을 종료합니다.')
#             break
#         elif choice == '1':
#             coin = coin-300 # 커피값 300원 차감
#             inventory_dict['change'] += 300
#             print(f'블랙 커피를 선택하셨습니다. 잔액: {coin}')
#             # 사용하는 재고) 커피 -30, 물 -100, 컵 -1
#             stock(inven=inventory_dict, coffee='using1', water='using4', 
#                   cup='using5', amount1=30, amount4=100, amount5=1)
#             show_stock()
#         elif choice == '2':
#             coin = coin-300 # 커피값 300원 차감
#             inventory_dict['change'] += 300
#             print(f'프림 커피를 선택하셨습니다. 잔액: {coin}')
#             # 사용하는 재고) 커피 -15, 프림 -15, 물 -100, 컵 -1
#             stock(inven=inventory_dict, coffee='using1', cream='using2', water='using4', 
#                   cup='using5', amount1=15, amount2=15, amount4=100, amount5=1)
#             show_stock()
#         elif choice == '3':
#             coin = coin-300 # 커피값 300원 차감
#             inventory_dict['change'] += 300
#             print(f'설탕 프림 커피를 선택하셨습니다. 잔액: {coin}')
#             # 사용하는 재고) 커피 -10, 프림 -10, 설탕 -10, 물 -100, 컵 -1
#             stock(inven=inventory_dict, coffee='using1', cream='using2', sugar='using3', water='using4', 
#                   cup='using5', amount1=15, amount2=15, amount3=10, amount4=100, amount5=1)
#             show_stock()
#         elif choice == '4':
#             show_stock()