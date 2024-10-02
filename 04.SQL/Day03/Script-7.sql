use sakila;

select 1 as num, 'abc' as str
union
select 'string' as num, 'xyz' as str;

desc customer;

desc actor;

# union 연산자(합집합) - 결합된 집합을 정렬하고 중복을 제거
select 'CUST' as type1, c.first_name, c.last_name
from customer as c
union all
select 'ACTR' as type1, a.first_name, a.last_name
from actor as a;

select count(first_name) from customer;
select count(first_name) from actor;

# union all: 중복되는 모든 값을 보여줌
select 'ACTR1' as type, a.first_name, a.last_name
from actor as a
union all
select 'ACTR2' as type, a.first_name, a.last_name
from actor as a;


select 'CUST' as type1, c.first_name, c.last_name
from customer as c
where c.first_name like 'J%' and c.last_name like 'D%'
union all 
select 'ACT' as type1, a.first_name, a.last_name
from actor as a
where a.first_name like 'J%' and a.last_name like 'D%';

# type1을 설정할 경우 union all과 동일
select 'CUST' as type1, c.first_name, c.last_name
from customer as c
where c.first_name like 'J%' and c.last_name like 'D%'
union
select 'ACT' as type1, a.first_name, a.last_name
from actor as a
where a.first_name like 'J%' and a.last_name like 'D%';

# union: 중복 제거
select c.first_name, c.last_name
from customer as c
where c.first_name like 'J%' and c.last_name like 'D%'
union
select a.first_name, a.last_name
from actor as a
where a.first_name like 'J%' and a.last_name like 'D%';

# intersect 연산자
select c.first_name, c.last_name
from customer as c
where c.first_name like 'D%' and c.last_name like 'T%'

select a.first_name, a.last_name
from actor as a
where a.first_name like 'D%' and a.last_name like 'T%';

# intersect 교집합이 없으므로 0 공집합
select c.first_name, c.last_name
from customer as c
where c.first_name like 'D%' and c.last_name like 'T%'
intersect
select a.first_name, a.last_name
from actor as a
where a.first_name like 'D%' and a.last_name like 'T%';


select c.first_name, c.last_name
from customer as c
where c.first_name like 'J%' and c.last_name like 'D%'
intersect
select a.first_name, a.last_name
from actor as a
where a.first_name like 'J%' and a.last_name like 'D%';

# inner join 연산자를 이용하여 공통 항목 검색 (intersect보다 확장성이 더 높음)
select c.first_name, c.last_name
from customer as c
inner join actor as a
on (c.first_name = a.first_name) and (c.last_name = a.last_name);

select c.first_name, c.last_name
from customer as c
where c.first_name like 'J%' and c.last_name like 'D%'
intersect
select a.first_name, a.last_name
from actor as a
where a.first_name like 'J%' and a.last_name like 'D%';

select c.first_name, c.last_name
from customer as c
	inner join actor as a
	on (c.first_name = a.first_name) and (c.last_name = a.last_name);
-- where a.first_name like 'J%' and a.last_name like 'D%';

# 차집합 (except)
select a.first_name, a.last_name
from actor as a
where a.first_name like 'J%' and a.last_name like 'D%'
except
select c.first_name, c.last_name
from customer as c
where c.first_name like 'J%' and c.last_name like 'D%';

# 복합 쿼리
select a.first_name, a.last_name
from actor as a
where a.first_name like 'J%' and a.last_name like 'D%'
union all
select a.first_name, a.last_name
from actor as a
where a.first_name like 'M%' and a.last_name like 'T%'
union
select c.first_name, c.last_name
from customer as c
where c.first_name like 'J%' and c.last_name like 'D%';

select first_name, last_name
from actor
where last_name like 'L%'
union
select first_name, last_name
from customer
where last_name like 'L%'
order by last_name;