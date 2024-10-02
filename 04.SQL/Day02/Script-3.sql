# 필터링

use sakila;

# 부등조건: <>, !=
select c.email, r.rental_date
from customer as c
	inner join rental as r
	on c.customer_id = r.customer_id
where date(r.rental_date) <> '2005-06-14';


# 범위조건: <, >, <=, >=
select customer_id, rental_date
from rental
where rental_date < '2005-05-25';

select customer_id, rental_date
from rental
where rental_date <= '2005-06-16'
	and rental_date >= '2005-06-14';
	

select customer_id, rental_date
from rental
where date(rental_date) <= '2005-06-16'
	and date(rental_date) >= '2005-06-14';

# between 연산자: between [하한] and [상한]
select customer_id, rental_date
from rental
where date(rental_date) between '2005-06-14' and '2005-06-16'; # 하한값, 상한값 위치가 바뀌면 결과 출력x

select customer_id, payment_date, amount
from payment
where amount between 10.0 and 11.99;

select last_name, first_name
from customer
where last_name between 'FA' and 'FRB';


select title, rating
from film
where rating='G' or rating='PG';

select title, rating
from film
where rating in ('G', 'PG');

# % 멤버십, PET%: PET로 시작하는 단어, %PET: PET로 끝나는 단어, %PET%: PET를 포함하는 단어
select title, rating
from film
where rating in (select rating from film where title like '%PET%');

select title, rating from film where title like '%PET%';


select last_name, first_name
from customer
where last_name like '_A_T%S';

select last_name, first_name
from customer
where last_name like 'Q%' or last_name like 'Y%';

# 정규표현식 ^[]
select last_name,	first_name
from customer
where last_name	REGEXP '^[QY]';

# is null
select rental_id, customer_id, return_date
from rental
where return_date is null;

# is not null
select rental_id, customer_id, return_date
from rental
where return_date is not null;


select payment_id, customer_id, amount, date(payment_date) as payment_date
from payment
where (payment_id between 101 and 120);


select payment_id, customer_id, amount, date(payment_date) payment_date
from payment
where (payment_id between 101 and 120)
and customer_id = 5 and not (amount > 6 or date(payment_date) = '2005-06-19');

select amount from payment
where amount in (1.98, 7.98, 9.98);




