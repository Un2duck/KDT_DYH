# ------------------------------------------------------------------------------
# 심사문제: 가장 낮은 점수, 높은 점수와 평균 점수를 구하는 함수 만들기
# 국어, 영어, 수학, 과학 점수 입력됨.
# 가장 높은 점수, 가장 낮은 점수, 평균 점수가 출력되게 만들기. (평균 점수는 실수 출력.)
# ------------------------------------------------------------------------------
# ex) 76 82 89 84
# >>> 낮은 점수: 76.00, 높은 점수: 89.00, 평균 점수: 82.75

# ex2) 89 92 73 83
# >>> 낮은 점수: 73.00, 높은 점수: 92.00, 평균 점수: 84.25

# scores=[76, 82, 89, 84]
# print(sum(scores)/len(scores))


# scores2=[89, 92, 73, 83]
# print(sum(scores2)/len(scores2))

korean, english, mathematics, science = map(int, input().split())

def get_min_max_score(*args):
    return max(args), min(args)

def get_average(**args):
    return sum(args.values())/len(args)

min_score, max_score = get_min_max_score(korean, english, mathematics, science)
average_score = get_average(korean=korean, english=english, mathematics=mathematics, science=science)
print('낮은 점수: {0:.2f}, 높은 점수: {1:.2f}, 평균 점수: {2:.2f}'.format(min_score, max_score, average_score))

min_score, max_score = get_min_max_score(english, science)
average_score = get_average(english=english, science=science)
print('낮은 점수: {0:.2f}, 높은 점수: {1:.2f}, 평균 점수: {2:.2f}'.format(min_score, max_score, average_score))