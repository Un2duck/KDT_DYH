'''
문제)
총 N개의 정수가 주어졌을 때, 정수 v가 몇 개인지 구하는 프로그램을 작성하시오.

입력)
첫째 줄에 정수의 개수 N(1 ≤ N ≤ 100)이 주어진다.
둘째 줄에는 정수가 공백으로 구분되어져있다.
셋째 줄에는 찾으려고 하는 정수 v가 주어진다.
입력으로 주어지는 정수와 v는 -100보다 크거나 같으며, 100보다 작거나 같다.

출력)
첫째 줄에 입력으로 주어진 N개의 정수 중에 v가 몇 개인지 출력한다.

ex)
입력)
11
1 4 1 2 4 2 4 2 3 4 4
2

출력)
3
'''
# 정수의 개수 받기.
n = int(input(''))
check = int(input(''))
import random
int_list = []
count = 0
for i in range(n):
    i = random.randint(1,10)
    int_list.append(i)
    if i == check:
        count += 1

for i in int_list:
    print(i, end=' ')
    
print(count)