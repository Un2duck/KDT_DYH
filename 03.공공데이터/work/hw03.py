'''
1. 대구광역시 전체 및 9개 구,군별 (중구, 동구, 서구, 남구, 북구, 수성구, 달서구, 달성군, 군위군) 
남녀 비율을 각각의 파이 차트로 구현하세요. (hw03.py)
- subplots를 이용하여 5x2 형태의 총 10개의 subplot을 파이 차트로 구현
- gender.csv 파일 사용

↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

실행 결과 (화면 출력)
대구광역시 : (남:1,162,046 여:1,205,137)
대구광역시 중구: (남:44,349 여:48,310)
대구광역시 동구: (남:168,160 여:175,088)
대구광역시 서구: (남:81,083 여:82,276)
대구광역시 남구: (남:65,312 여:71,581)
대구광역시 북구: (남:206,403 여:210,039)
대구광역시 수성구: (남:196,513 여:211,798)
대구광역시 달서구: (남:256,966 여:267,318)
대구광역시 달성군: (남:131,520 여:127,760)
대구광역시 군위군: (남:11,740 여:10,967)

'''
# 모듈 생성 및 파일 로딩
# ----------------------------------------------------------------------------------------------
import csv
import matplotlib.pyplot as plt
import koreanize_matplotlib

f = open('gender.csv', encoding='euc-kr')
data = csv.reader(f)

# 대구 정보
# ----------------------------------------------------------------------------------------------
daegu = ['대구광역시'] * 10
gu_list = ['','중구','동구','서구','남구','북구','수성구','달서구','달성군','군위군']
daegu_gu_list = [daegu[i] + " " + gu_list[i] for i in range(len(gu_list))]
# ----------------------------------------------------------------------------------------------

# # 선 그래프로 그리기

# for gu in daegu_gu_list:
#     male_list = []
#     female_list = []
#     for row in data:
#         if gu in row[0]:
#             for i in range(106, 207):
#                 male_list.append(int(row[i].replace(',','')))
#                 female_list.append(int(row[i+103].replace(',','')))
#             break
    
#     color = ['cornflowerblue', 'tomato']
#     plt.plot(male_list, label='남성', color=color[0])
#     plt.plot(female_list, label='여성', color=color[1])
#     plt.title(gu + ' 남녀 인구수 비교')
#     plt.xlabel('나이')
#     plt.ylabel('인구수')
#     plt.legend()
#     plt.show()

# 파이 그래프로 그리기

# population = [] # Pie chart에 넣을 데이터 (남, 여 인구수)

# for gu in daegu_gu_list:
#     male_total = 0
#     female_total = 0
#     male_list = []
#     female_list = []
#     for row in data:
#         if gu in row[0]:
#             for i in range(106, 207):
#                 male_list.append(int(row[i].replace(',','')))
#                 female_list.append(int(row[i+103].replace(',','')))
#                 # male_total: 남자 총 인구 / # female_total: 여자 총 인구
#                 male_total = sum(male_list)
#                 female_total = sum(female_list)
#             break
#     print(f'{gu}: (남:{male_total:,}명, 여:{female_total:,}명)')

#     population = [male_total, female_total]
#     color = ['cornflowerblue', 'tomato']
#     plt.pie(population, labels=['남성', '여성'], autopct='%.1f%%', colors=color, startangle=90)
#     plt.title(gu + '남녀 성별 비율')
#     plt.show()

# 파이 그래프로 그리기 (subplot 이용하기)

population = [] # Pie chart에 넣을 데이터 (남, 여 인구수)
population_list = []
for gu in daegu_gu_list:
    male_total = 0
    female_total = 0
    male_list = []
    female_list = []
    for row in data:
        if gu in row[0]:
            for i in range(106, 207):
                male_list.append(int(row[i].replace(',','')))
                female_list.append(int(row[i+103].replace(',','')))
                # male_total: 남자 총 인구 / # female_total: 여자 총 인구
                male_total = sum(male_list)
                female_total = sum(female_list)
            break
    print(f'{gu}: (남:{male_total:,}명, 여:{female_total:,}명)')

    population = [male_total, female_total]
    color = ['cornflowerblue', 'tomato']

    population_list.append(population)

# 그래프 담을 창 생성
fig=plt.figure(figsize=(10,10))

# 그래프 담을 공간 생성 axes
axes = fig.subplots(5,2)

cnt=0
for i in range(5):
    for j in range(2):
        axes[i,j].pie(population_list[cnt], labels=['남성', '여성'], autopct='%.1f%%', startangle=90)
        axes[i,j].set_title(daegu_gu_list[cnt])
        cnt+=1

plt.suptitle('대구광역시 구별 남녀 인구비율')
plt.show()