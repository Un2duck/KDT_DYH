import csv

f = open('subwayfee.csv', encoding='utf-8-sig')
data = csv.reader(f)
header = next(data) # 한 줄을 읽고 다음 줄로 이동
max_rate = 0
rate = 0
max_row = []
max_total_num = 0

for row in data:
    for i in range(4, 8):
        row[i] = int(row[i]) # 4, 5, 6, 7 컬럼 값을 정수로 변환
    total_count = row[4] + row[6] # 유임승차수 + 무임승차수
    
    if (row[6] != 0) and (total_count > 100000):
        rate = row[4] / total_count
        if rate > max_rate :
            max_rate = rate
            max_row = row
            max_total_num = total_count
            print(f'역이름: {max_row[3]}, 전체인원: {max_total_num:,}명,'
                  f'유임승차인원: {max_row[4]:,}명, '
                  f'유임승차 비율: {round(max_rate * 100, 1):,}%')
print('-'*80)
print('최대 유임 승차역')
print(f'역이름: {max_row[3]}, 전체인원 {max_total_num:,}명, 0'
      f'유임승차인원: {max_row[4]:,}명, '
      f'유임승차 비율: {round(max_rate * 100, 1):,}%')
f.close()