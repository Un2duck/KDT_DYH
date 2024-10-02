import pymysql
import pandas as pd
import csv

conn = pymysql.connect(host='localhost', user='root', password='1234',
                       db='sakila', charset='utf8')
# cur = conn.cursor()
cur = conn.cursor(pymysql.cursors.DictCursor)
cur.execute('select * from language')

rows= cur.fetchall() # 모든 데이터를 가져옴
print(rows)

language_df = pd.DataFrame(rows)
print(language_df)
print()
print(language_df['name'])
cur.close()
conn.close() # 데이터베이스 연결 종료