# ------------------------------------------------------------------------------
# 심사문제 : 특정 단어 개수 세기
# 입력된 문자열에서 'the'의 개수를 출력하는 프로그램 만들기
# 단, 모든 문자가 소문자인 'the'만 찾으면 되며 'them', 'there', ' their'등은 포함x
# ------------------------------------------------------------------------------

# The grown-ups' response, this time, was to advise me to lay aside my drawings of boa constrictors,
# whether from the inside or the outside, and devote myself instead to geography, history, arithmetic, and grammar. 
# That is why, at the age of six, I gave up what might have been a magnificent career as a painter.
# I had been disheartened by the failure of my Drawing Number One and my Drawing Number Two.
# Grown-ups never understand anything by themselves, and it is tiresome for children to be always and forever explaining things to them.



import string

# strings ="The grown-ups' response, this time, was to advise me to lay aside my drawings of boa constrictors, \
#     whether from the inside or the outside, and devote myself instead to geography, history, arithmetic, and grammar. \
#     That is why, at the age of six, I gave up what might have been a magnificent career as a painter. \
#     I had been disheartened by the failure of my Drawing Number One and my Drawing Number Two. \
#     Grown-ups never understand anything by themselves, \
#     and it is tiresome for children to be always and forever explaining things to them."

strings=input('')
strings=strings.split()

count_the = 0
for str in strings:
    if str == 'the':
        count_the += 1

print(strings)
print(count_the)