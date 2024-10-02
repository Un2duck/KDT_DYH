import pymysql
import pandas as pd

conn = pymysql.connect(host='localhost', user='root', password='1234',
                       db='sqlclass_db', charset='utf8')
cur = conn.cursor()
# cur = conn.cursor(pymysql.cursors.DictCursor)
cur.execute('''select t.title, t.edition_number, t.copyright 
            from titles as t
            where t.copyright >= '2013'
            order by t.copyright desc;
            ''')

rows= cur.fetchall() # 모든 데이터를 가져옴
print(rows)

author_df = pd.DataFrame(rows)
print(author_df)
print()

cur.close()
conn.close() # 데이터베이스 연결 종료