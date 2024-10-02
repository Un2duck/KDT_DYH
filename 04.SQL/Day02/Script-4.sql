use sakila;

# 교차 조인
select c.first_name, c.last_name, a.address
from customer as c cross join address as a;

select count(*) from customer;
select count(*) from address;
select count(*) from customer as c join address as a;

# 내부 조인 (inner join)
select c.first_name, c.last_name, a.address, a.district
from customer as c inner join address as a
on c.address_id = a.address_id;

select count(*)
from customer as c inner join address as a
on c.address_id = a.address_id;

# ANSI join 문법
# 이전 문법
select c.first_name, c.last_name, a.address, a.postal_code
from customer as c join address as a
where c.address_id = a.address_id and a.postal_code = 52137;

# SQL92 문법
select c.first_name, c.last_name, a.address, a.postal_code
from customer as c join address as a
	on c.address_id = a.address_id
where a.postal_code = 52137;

desc city;

# 세 개 이상 테이블 조인
select c.first_name, c.last_name, ct.city, a.address, a.district, a.postal_code
from customer as c
	inner join address as a on c.address_id = a.address_id
	inner join city as ct on a.city_id =ct.city_id;
	
# 서브 쿼리 사용
select c.first_name, c.last_name, addr.address, addr.city, addr.district
from customer as c
	inner join
	# 서브 쿼리 별칭: addr
	(select a.address_id, a.address, ct.city, a.district
	from address as a
		inner join city as ct
		on a.city_id = ct.city_id
	where a.district = 'California'
	) as addr
on c.address_id = addr.address_id;


select f.title
from film as f
	inner join film_actor as fa
	on f.film_id = fa.film_id
	inner join actor as a
	on fa.actor_id = a.actor_id
where ((a.first_name = 'CATE' and a.last_name = 'MCQUEEN')
	or (a.first_name = 'CUBA' and a.last_name = 'BIRCH'));
	

# CATE MCQUEEN만 검색
select f.title, f.film_id, a1.first_name, a1.last_name 
from film as f
	inner join film_actor as fa1
	on f.film_id = fa1.film_id
	inner join actor a1
	on fa1.actor_id = a1.actor_id
where (a1.first_name = 'CATE' and a1.last_name = 'MCQUEEN');

# CUBA BIRCH만 검색
select f.title, f.film_id, a2.first_name, a2.last_name 
from film as f
	inner join film_actor as fa2
	on f.film_id = fa2.film_id
	inner join actor a2
	on fa2.actor_id = a2.actor_id
where (a2.first_name = 'CUBA' and a2.last_name = 'BIRCH');

select f.title
from film as f
	inner join film_actor as fa1
	on f.film_id = fa1.film_id
	inner join actor a1
	on fa1.actor_id = a1.actor_id
	inner join film_actor as fa2
	on f.film_id = fa2.film_id
	inner join actor a2
	on fa2.actor_id = a2.actor_id
where (a1.first_name = 'CATE' and a1.last_name = 'MCQUEEN')
	and (a2.first_name = 'CUBA' and a2.last_name = 'BIRCH');
	
	