import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

conn = pymysql.connect(host='localhost', user='root', password='1234',
                       db='sqlteam4_db', charset='utf8')
# cur = conn.cursor()
cur1 = conn.cursor(pymysql.cursors.DictCursor)
cur2 = conn.cursor(pymysql.cursors.DictCursor)

query1 = '''
select y15.city as '연도별 지하철 교통카드 요금', y15.subcardfee as '2015', y16.subcardfee as '2016', y17.subcardfee as '2017', y18.subcardfee as '2018', y19.subcardfee as '2019', y20.subcardfee as '2020', y21.subcardfee as '2021', y22.subcardfee as '2022', y23.subcardfee as '2023', y24.subcardfee as '2024'
from y2015 as y15
	inner join y2016 as y16
	on y15.city = y16.city
	inner join y2017 as y17
	on y16.city = y17.city
	inner join y2018 as y18
	on y17.city = y18.city
	inner join y2019 as y19
	on y18.city = y19.city
	inner join y2020 as y20
	on y19.city = y20.city
	inner join y2021 as y21
	on y20.city = y21.city
	inner join y2022 as y22
	on y21.city = y22.city
	inner join y2023 as y23
	on y22.city = y23.city
	inner join y2024 as y24
	on y23.city = y24.city
where y15.city in ('서울','인천','대구','부산','대전','광주');
'''

query2='''
select y15.city as '연도별 지하철 현금 요금', y15.subcashfee as '2015', y16.subcashfee as '2016', y17.subcashfee as '2017', y18.subcashfee as '2018', y19.subcashfee as '2019', y20.subcashfee as '2020', y21.subcashfee as '2021', y22.subcashfee as '2022', y23.subcashfee as '2023', y24.subcashfee as '2024'
from y2015 as y15
	inner join y2016 as y16
	on y15.city = y16.city
	inner join y2017 as y17
	on y16.city = y17.city
	inner join y2018 as y18
	on y17.city = y18.city
	inner join y2019 as y19
	on y18.city = y19.city
	inner join y2020 as y20
	on y19.city = y20.city
	inner join y2021 as y21
	on y20.city = y21.city
	inner join y2022 as y22
	on y21.city = y22.city
	inner join y2023 as y23
	on y22.city = y23.city
	inner join y2024 as y24
	on y23.city = y24.city
where y15.city in ('서울','인천','대구','부산','대전','광주');
'''

cur1.execute(query1)
cur2.execute(query2)

rows1= cur1.fetchall() # 모든 데이터를 가져옴
rows2= cur2.fetchall() # 모든 데이터를 가져옴

subcardfee_df = pd.DataFrame(rows1)
subcashfee_df = pd.DataFrame(rows2)

cur1.close()
cur2.close()
conn.close() # 데이터베이스 연결 종료

# print(subcardfee_df.columns.tolist()[1:])
# print(subcardfee_df.values)
# print(subcardfee_df.values.tolist())

card_title_list = []
for i in range(6):
	card_title_list.append(subcardfee_df.values.tolist()[i][0])

card_y = []
for i in subcardfee_df.values.tolist():
	card_y.append(list(map(int,i[1:])))

# for x in range(6):
# 	plt.bar(subcardfee_df.columns.tolist()[1:], card_y[x])
# 	plt.title(card_title_list[x])
# 	plt.show()

cash_title_list = []
for i in range(6):
	cash_title_list.append(subcashfee_df.values.tolist()[i][0])

cash_y = []
for i in subcardfee_df.values.tolist():
	cash_y.append(list(map(int,i[1:])))

for x in range(6):
	plt.bar(subcashfee_df.columns.tolist()[1:], cash_y[x])
	plt.title(cash_title_list[x])
	plt.show()