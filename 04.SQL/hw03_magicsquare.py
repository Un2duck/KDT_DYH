'''
§ 마방진 원리
• 시작 위치: 첫 행의 가운데 열에서 시작함(y1=0,	x1=1)
§ 이동 규칙
• 1) 다음 위치는 오른쪽 대각선 방향으로 이동 (y2=y1-1,	x2=x1+1)
• 2) y축 방향으로 범위가 벗어난 경우, y는 마지막 행(size-1)으로 이동
• 3) x축 방향으로 범위가 벗어난 경우, x는 첫 번째 열(0)으로 이동
• 4) 다음 이동 위치에 이미 값이 있는 경우, y는 y	+ 1
'''


# ------------------------------------------------------------------------------------
# 함수 만들기
# ------------------------------------------------------------------------------------

# # 마방진 만들기
# def make_matrix(number):
#     global matrix
#     global rows
#     global cols
#     rows = number
#     cols = number
#     matrix = [[0 for col in range(number)] for row in range(number)]
#     # [[0] * size for i in range(number)] 와 동일

# # 숫자 만들기
# def make_numbers(number):
#     global new_numbers
#     new_numbers = [i+1 for i in range(number**2)]


# # 규칙만들기
# def rule():
#     global row_ax
#     global col_ax

#     matrix[0][cols//2] = new_numbers[0]
#     row_ax = 0 # :0 시작 줄은 항상 1번줄
#     col_ax = cols//2 # :1 시작 열은 항상 가운데줄

#    # 1번 규칙
#     try:
#         if matrix[row_ax-1][col_ax+1] == 0:
#             matrix[row_ax-1][col_ax+1] = 2
#             row_ax += 1 # 1
#             col_ax += 1 # 2
#     # 2번 규칙
#     except:
#             matrix[row_ax-1+rows][col_ax+1] = 2
#             row_ax += 1 # 2
#             col_ax = cols-1 # 2

#     # # 3번 규칙
#     try:
#         if matrix[row_ax-1][col_ax+1] == 0:
#             matrix[row_ax-1][col_ax+1] = 3
#     except:
#             matrix[row_ax-1][col_ax-2] = 3

# # ------------------------------------------------------------------------------------

# make_matrix(3)
# make_numbers(3)

# # 조건만들기
# rule()

# print(matrix)
# print(row_ax, col_ax)

# ------------------------------------------------------------------------------------

def make_magic():
    row = 0
    col = size//2
    matrix = [[0 for col in range(size)] for row in range(size)]
    matrix[row][col] = 1

    for i in range(2, size*size+1):
        check_row = row-1 if row-1 >= 0 else size-1
        check_col = col+1 if col+1 <= size-1 else 0
        if matrix[check_row][check_col] == 0:
            row = check_row
            col = check_col
            matrix[row][col] = i
        else:
            check_same = row+1 if row+1 <= size-1 else 0
            row = check_same
            matrix[row][col] = i

    for i in matrix:
        print(i)


while True:
    size = int(input('홀수차 배열의 크기를 입력하세요: '))
    if size % 2 == 0:
        print('짝수를 입력하였습니다. 다시 입력하세요.')
        pass
    else:
        make_magic()
        break