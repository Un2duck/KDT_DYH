# ------------------------------------------------------------------------------
# 심사문제 : 높은 가격순으로 출력하기
# 물품 가격 여러개가 문자열 한 줄로 입력되고, 각 가격은 ;(세미콜론)으로 구분.
# 입력된 가격을 높은 가격순으로 출력하는 프로그램 만들기.
# 가격은 길이 9로 만든 뒤 오른쪽으로 정렬하고 천단위로 ,(콤마)를 넣기.
# ------------------------------------------------------------------------------

# ex) 51900;83000;158000;367500;250000;59200;128500;1304000
# >>> 1,304,000
#       367,500
#       250,000
#       158,000
#       128,500
#        83,000
#        59,200
#        51,900

# prices='51900;83000;158000;367500;250000;59200;128500;1304000'

prices=input('')
prices=prices.split(';')
prices=list(map(int, prices))
prices.sort(reverse=True)
for price in prices:
    print('{0:>9,}'.format(price))