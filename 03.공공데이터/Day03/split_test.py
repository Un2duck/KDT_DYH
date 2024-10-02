'''
    str.split 테스트
'''

temp_string = '1;2,3:456/789'

print(temp_string.split(';,')) # (;,)를 단일 구분자로 취급. 즉, 구분자 없다고 인식.
print(len(temp_string.split(';,')))

print(temp_string.split(',')) # 하나의 구분자만 가능
print(len(temp_string.split(',')))

city = '대구광역시 북구 산격3동(2723063000)'
print(city.split('('))