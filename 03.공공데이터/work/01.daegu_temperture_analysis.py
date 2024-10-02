'''
1. 대구 기온 데이터에서 시작 연도, 마지막 연도를 입력하고 특정 월의 최고 기온 및 최저
기온의 평균값을 구하고 그래프로 표현
    * daegu-utf8.csv 또는 daegu-utf8-df.csv 파일 이용 (완료)
    * 데이터 구조 (완료)
        ['날짜', '지점', '평균기온', '최저기온', '최고기온’]
        [0] [1] [2] [3] [4]
    * 화면에서 측정할 달을 입력 받아서 진행 (완료)
    
    * 해당 기간 동안 최고기온 평균값 및 최저기온 평균값 계산 (완료)
        - 최고기온 및 최저기온 데이터를 이용하여 입력된 달의 각각 평균값을 구함
        - 문자열 형태의 ‘날짜’ 열의 데이터는 datetime 으로 변경함: (완료)
    * 하나의 그래프 안에 2개의 꺾은선 그래프로 결과를 출력
        - 마이너스 기호 출력 깨짐 방지
        - 입력된 월을 이용하여 그래프의 타이틀 내용 변경
        - 최고 온도는 빨간색, 최저 온도는 파란색으로 표시하고 각각 마커 및 legend를 표시
'''

# 모듈 생성
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib


# daegu-utf8.csv 또는 daegu-utf8-df.csv 파일 이용
daegu_weather_df = pd.read_csv('daegu-utf8-df.csv', encoding='utf-8-sig')
#daegu_weather_df = pd.read_csv(r'C:\Users\KDP-50\OneDrive\바탕 화면\공공데이터\Day01\daegu-utf8.csv', encoding='utf-8-sig')


# 데이터 구조 => ['날짜', '지점', '평균기온', '최저기온', '최고기온’]
daegu_weather_df.columns = ['날짜', '지점', '평균기온', '최저기온', '최고기온']

# 화면에서 측정할 달을 입력 받아서 진행
start_year = int(input('시작 연도를 입력하세요: '))
end_year= int(input('마지막 연도를 입력하세요: '))
search_month = int(input('기온 변화를 측정할 달을 입력하세요: '))

# (추가) 결측치 정리
daegu_weather_df = daegu_weather_df.dropna(axis=0)

# 해당 기간 동안 최고기온 평균값 및 최저기온 평균값 계산
daegu_weather_df['날짜'] = pd.to_datetime(daegu_weather_df['날짜'], format='%Y-%m-%d')

def draw_two_plots(title, x_data, start_max_temp_list, label_y1, start_min_temp_list, label_y2):
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 4))
    plt.plot(x_data, start_max_temp_list, marker='s', markersize=6, color='r', label=label_y1)
    plt.plot(x_data, start_min_temp_list, marker='s', markersize=6, color='b', label=label_y2)

    plt.title(title)
    plt.legend(loc=2)
    plt.show()

def main():
    start_year_df = daegu_weather_df[daegu_weather_df['날짜'].dt.year == start_year]
    # month_df = start_year_df[start_year_df['날짜'].dt.month == search_month]

    start_max_temp_list = [0] * (end_year-start_year)
    start_min_temp_list = [0] * (end_year-start_year)

    for year in range(end_year-start_year):
        start_year_df = daegu_weather_df[(daegu_weather_df['날짜'].dt.year == start_year + year) & 
                                        (daegu_weather_df['날짜'].dt.month == search_month)]
        start_max_temp_list[year] = round(start_year_df['최고기온'].mean(), 1)
        start_min_temp_list[year] = round(start_year_df['최저기온'].mean(), 1)

    # start_high_temp_mean = round(sum(start_max_temp_list) / len(start_max_temp_list), 1)
    # start_low_temp_mean = round(sum(start_min_temp_list) / len(start_min_temp_list), 1)

    print(f'{search_month}월 최고기온 평균:\n{start_max_temp_list}')

    print(f'{search_month}월 최저기온 평균:\n{start_min_temp_list}')

    x_data = [i+start_year for i in range(end_year-start_year)]
    draw_two_plots(f'{start_year}년부터 {end_year}년까지 {search_month}월의 기온 변화', x_data,
                    start_max_temp_list, '최고기온',
                    start_min_temp_list, '최저기온')
    
main()