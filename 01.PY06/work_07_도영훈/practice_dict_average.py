# -----------------------------------------------------------------------------------------------
# 평균 점수가 출력되게 만들기.
# -----------------------------------------------------------------------------------------------
# maria = {'korean':94, 'english':91, 'mathematics':89, 'science':83}

# print(average)
# >>> 89.25

maria = {'korean':94, 'english':91, 'mathematics':89, 'science':83}
b = maria.values()
average=sum(b)/len(b)
print(average)
