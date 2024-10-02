'''
1. 지하철 각 노선별 최대 하차 인원을 막대그래프로 표시하고, 하차인원 출력
 - 출근 시간대: 07:00~08:59
 - 사용 파일: subwaytime.csv	또는 subway.xls
    - 07:00~07:59 하차: index[11],	08:00~08:59 하차: index	[13]
 - 각 지하철 노선별 가장 많이 내리는 지하철 역 분석
    - 1 호선, 2 호선, 3 호선, 4 호선, 5 호선, 6 호선, 7 호선
 - 하차 인원은 1,000 단위로 콤마를 찍어서 구분할 것
 - 7개의 지하철 역을 막대 그래프로 표시
 - Bar chart 의 x 축은 (노선 + 지하철 역 이름)을 표시하고, y 축은 인원수를 표시
'''

# 1. 출근 시간대 (07:00~08:59)의 전체 하차인원 구하기
# 2. 해당 시간대에 1호선 ~ 7호선별 가장 많이 내리는 지하철 역 구하기 (하차인원)

# 1호선부터 구하기.
# 1호선 ~ 7호선 리스트 만들기

import csv
import matplotlib.pyplot as plt
import koreanize_matplotlib

subway_list = ['1호선', '2호선', '3호선', '4호선', '5호선', '6호선', '7호선']

def make_subway(number):
    with open('subwaytime.csv', encoding='utf-8-sig') as f:
        data = csv.reader(f)
        next(data)
        next(data)

        station_list = []
        most_subway_list = []
        passenger_list = []

        max_num = -1
        for row in data:
            row[4:] = map(int, row[4:])
            
        # #   해당시간대 1호선의 하차인원이 가장 많은 역 찾기.
            if row[1] == subway_list[number]:
                if (row[11]+row[13]) > max_num:
                    max_num = (row[11]+row[13])
                    passenger_num =	sum(row[11:14:2])
                    station_name = row[1] + ' 최대 하차역: ' + row[3] + '역'
                    station_list.append((station_name, passenger_num))

        sorted_passenger_list = sorted(station_list, key=lambda x:x[1], reverse=True)
        station_name, station_passenger = zip(*sorted_passenger_list[:10])

        most_subway_list.append(station_name[0])
        passenger_list.append(station_passenger[0])
        print(f'출근시간대 {most_subway_list[0]}, 하차인원: {passenger_list[0]:,}명')

        return [most_subway_list[0], passenger_list[0]]

test1 = []
test2 = []
for i in range(7):
    test1.append(make_subway(i)[0])
    test2.append(make_subway(i)[1])
print(test1)
print(test2)

# -------------------------------------------------------------------------------
# 그래프 그리기

plt.figure(figsize=(5,5))
plt.bar(test1, test2)
plt.title('출근 시간대 지하철 노선별 최대 하차 인원 및 하차역', size=12)
plt.xticks(test1, rotation=70)
plt.tight_layout()
plt.show()