# ------------------------------------------------------------------------------
# [실습] 10번 숫자 데이터 입력을 받습니다.
#       - 숫자 데이터를 모두 더해서 합계가 30 이상이 되면,
#       10번 입력을 받지 않았어도 종료.
# ------------------------------------------------------------------------------
# ex) 5 15 20 -> 40 // 5 5 5 5 5 5 -> 30 // 1 1 1 1 1 1 1 1 1 29 -> 38

total=0
nums=0
for _ in range(10):
    nums = int(input('숫자를 입력하세요.'))
    total = total + nums
    if total >= 30:
        break
    print(f'합계는 {total}입니다.')
print(f'최종 합계는 {total}입니다.')

    # for i in range(len(nums)):
    #     total=total+i
    #     if total>=30:
    #         break