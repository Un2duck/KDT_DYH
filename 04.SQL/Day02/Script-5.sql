use sqlclass_db;
create table customer
	(customer_id smallint unsigned,
	first_name varchar(20),
	last_name varchar(20),
	birth_date date,
	spouse_id smallint unsigned,
	constraint primary key (customer_id));
	
desc customer;

insert into customer (customer_id, first_name, last_name, birth_date, spouse_id)
values
(1, 'John', 'Mayer', '1983-05-12', 2),
(2, 'Mary', 'Mayer', '1990-07-30', 1),
(3, 'Lisa', 'Ross', '1989-04-15', 5),
(4, 'Anna', 'Timothy', '1988-12-26', 6),
(5, 'Tim', 'Ross', '1957-08-15', 3),
(6, 'Steve', 'Donell', '1967-07-09', 4);

insert into customer (customer_id, first_name, last_name, birth_date)
values
(7, 'Donna', 'Trapp', '1978-06-23');

select * from customer;


# 셀프 조인 (self-join)
select
	cust.customer_id,
	cust.first_name,
	cust.last_name,
	cust.birth_date,
	cust.spouse_id,
	spouse.first_name as spouse_firstname,
	spouse.last_name as spouse_lastname
from customer as cust
	join customer as spouse
	on cust.spouse_id = spouse.customer_id;
	
