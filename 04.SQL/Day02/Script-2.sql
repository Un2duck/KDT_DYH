use sakila;
select * from language;

select name, last_update from language;

select name from language;

# select절에 추가할 수 있는 항목
select language_id,
	'COMMON' as language_usage,
	language_id * 3.14 as lang_pi_value,
	upper(name) as language_name
from language;

select actor_id from film_actor order by actor_id;

# 중복 제거
select distinct actor_id from film_actor order by actor_id;

# 임시 테이블 (휘발성의 테이블: 데이터베이스 세션이 닫힐 때 사라짐)
create temporary table actors_j
	(actor_id smallint(5),
	first_name varchar(45),
	last_name varchar(45));

desc actors_j;

select concat()
from
	(select first_name, last_name, email
	from customer
	where first_name = 'JESSIE')
	as cust;

insert into actors_j
	select actor_id, first_name, last_name
	from actor where last_name like 'J%';

select * from actors_j;

# 가상 테이블
create view cust_vw as
	select customer_id, first_name, last_name, active
	from customer;

select * from cust_vw;

select first_name, last_name
from cust_vw where active=0;

# join(INNER JOIN)
select customer.first_name, customer.last_name,
	time(rental.rental_date) as rental_time
from customer inner join rental
	# 연결(결합) 조건: on
	on customer.customer_id = rental.customer_id
where date(rental.rental_date) = '2005-06-14';

# date 함수 / time 함수
select date('2021-07-29 09:02:03');
select time('2021-07-29 09:02:03');


select title, rating, rental_duration
from film
where rating='G' and rental_duration >= 7;

select title, rating, rental_duration
from film
where (rating='G' and rental_duration >= 7) or (rating='PG-13' and rental_duration < 4);


select c.first_name, c.last_name, count(*) as rental_count
from customer as c
	inner join rental as r
	on c.customer_id = r.customer_id
group by c.first_name, c.last_name
having count(*) >= 40
order by count(*) desc;

# order by (asc:default) 오름차순 / desc 내림차순
select c.first_name, c.last_name, 
	time(r.rental_date) as rental_time
from customer as c inner join rental as r
	on c.customer_id = r.customer_id
where date(r.rental_date) = '2005-06-14'
order by c.last_name, c.first_name asc;

select c.first_name, c.last_name,
	time(r.rental_date) as rental_time
from customer as c inner join rental as r
	on c.customer_id = r.customer_id
where date(r.rental_date) = '2005-06-14'
order by time(r.rental_date) desc;


select c.first_name, c.last_name,
	time(r.rental_date) as rental_time
from customer as c 
	inner join rental as r
	on c.customer_id = r.customer_id
where date(r.rental_date) = '2005-06-14'
order by 1 desc;

# actor	테이블에서 모든 배우의 actor_id, first_name, last_name을 검색하고 last_name, first_name을 기준으로 오름차순 정렬
select actor_id, first_name, last_name
from actor
order by last_name, first_name;

# 성이 ‘WILLIAMS’ 또는 ‘DAVIS’인 모든 배우의 actor_id,	first_name,	last_name을 검색
select actor_id, first_name, last_name
from actor
where last_name = 'WILLIAMS' or last_name = 'DAVIS';

# rental 테이블에서 2005년 7월 5일 영화를 대여한 고객 ID를 반환하는 쿼리를 작성하고, date()함수로 시간요소를 무시
select distinct customer_id
from rental
where date(rental_date) = '2005-07-05';


select c.store_id, c.email, r.rental_date, r.return_date
from customer as c inner join rental as r
	on c.customer_id = r.customer_id
where date(r.rental_date) = '2005-06-14'
order by return_date desc;