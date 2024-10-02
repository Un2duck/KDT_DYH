class Student:
    def __init__(self, name, grade, number):
        self.name = name
        self.grade = grade
        self.number = number

    def __repr__(self):
        return f'({self.name}, {self.grade}, {self.number})'
    
# Student 객체 리스트 생성
students = [Student('홍길동', 3.9, 20240303),
            Student('김유신', 3.0, 20240302),
            Student('박문수', 4.3, 20240301)]

print(students[0])

print('-'*50)
print('Student 객체에서 name 기준으로 오름차순 정렬')
sorted_list = sorted(students, key=lambda s: s.name)
print(sorted_list)

print('-'*50)
print('Student 객체에서 grade 기준으로 내림차순 정렬')
sorted_grade_list = sorted(students, key=lambda s: s.grade, reverse=True)
print(sorted_grade_list)