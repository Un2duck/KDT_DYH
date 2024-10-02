# ------------------------------------------------------------------------------
# 연습문제: 파일 경로에서 파일명만 가져오기
# ------------------------------------------------------------------------------

path= 'C:\\Users\\dojang\\AppData\\Local\\Programs\\Python\\Python36-32\\python.exe'

file = path.split('\\')
filename=file[-1]
print(filename)

file = path.split('\\')
file.reverse()
filename=file[0]
print(filename)