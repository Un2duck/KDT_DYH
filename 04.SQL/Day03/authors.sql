use sqlclass_db;

DROP TABLE IF exists authors;
create table authors
	(author_id int, 
	firstname varchar(20),
	lastname varchar(30),
	primary key (author_id)
	);
	

insert into authors
	(author_id, firstname, lastname)
	values 
	(1, 'Paul', 'Deitel'),
	(2, 'Harvey', 'Deitel'),
	(3, 'Abbey', 'Deitel'),
	(4, 'Dan', 'Quirk'),
	(5, 'Michael', 'Morgano');

delete from authors;

select *
from authors;

DROP TABLE IF exists titles;
create table titles
	(isbn VARCHAR(20),
	title VARCHAR(100),
	edition_number INT,
	copyright VARCHAR(4),
	primary key (isbn)
	);
	
insert into titles
	(isbn, title, edition_number, copyright)
	values
	('0132151006', 'Internet & World Wide Web	How to Program', 5, '2012'),
	('0133807800', 'Java How to Program', 10, '2015'),
	('0132575655', 'Java How to Program, LateObjects Version', 10, '2015'),
	('013299044X', 'C How to Program', 7, '2013'),
	('0132990601', 'Simply Visual Basic 2010', 4, '2013'),
	('0133406954', 'Visual Basic 2012 How to Program', 6, '2014'),
	('0133379337', 'Visual C# 2012 How to Program', 5, '2014'),
	('0136151574', 'Visual C++ How to Program', 2, '2008'),
	('0133378713', 'C++ How to Program', 9, '2014'),
	('0133764036', 'Android How to Program', 2, '2015'),
	('0133570924', 'Android for Programmers: An App-Driven Approach, Volume 1', 2, '2014'),
	('0132121360', 'Android for Programmers: An App-Driven Approach', 1, '2012');
	
select *
from titles;

delete from titles;

select *
from author_isbn;

delete from author_isbn;

DROP TABLE IF exists author_isbn;
create table author_isbn
	(author_id int,
	isbn varchar(20),
	foreign key (author_id) references authors(author_id),
	foreign key (isbn) references titles(isbn)
	);
	
delete from author_isbn;
insert into author_isbn (author_id, isbn)
	values
	(1, '0132151006'),
	(2, '0132151006'),
	(3, '0133807800'),
	(1, '0132575655'),
	(2, '013299044X'),
	(1, '013299044X'),
	(2, '0132575655'),
	(1, '013299044X'),
	(2, '013299044X'),
	(1, '0132990601'),
	(2, '0132990601'),
	(3, '0132990601'),
	(1, '0133406954'),
	(2, '0133406954'),
	(3, '0133406954'),
	(1, '0133379337'),
	(2, '0133379337'),
	(1, '0136151574'),
	(2, '0136151574'),
	(4, '0136151574'),
	(1, '0133378713'),
	(2, '0133378713'),	
	(1, '0133764036'),
	(2, '0133764036'),
	(3, '0133764036'),
	(1, '0133570924'),
	(2, '0133570924'),
	(3, '0133570924'),
	(1, '0132121360'),
	(2, '0132121360'),
	(3, '0132121360'),
	(5, '0132121360');
	

# 문제 1. 저작권 2013년 이후 도서 출력
# titles 테이블에서 copyright가 2013년 이후의 책 정보를 정렬 후 출력
# title, edition_number, copyright 필드를 copyright 필드의 내림차순으로 정렬하여 출력
# 2013년 포함

select t.title, t.edition_number, t.copyright
from titles as t
order by t.copyright desc
where year(t.copyright) >= '2013';

# 문제 2. ‘D’로 시작하는 저자 이름 출력
# authors 테이블에서 lastname이 ‘D’로 시작하는 저자의 id, firstname, lastname 출력
select a.author_id, a.firstname, a.lastname
from authors as a
where a.lastname like 'D%';

# 문제 3. 저자 이름의 두 번째 글자가 ‘o’를 포함하는 저자 이름 출력
# authors 테이블에서 저자의 lastname의 두 번째 글자에 ‘o’를 포함하는 저자 정보 출력
select *
from authors as a
where a.lastname like '_o%';

# 문제 4. 저자 이름을 오름차순으로 정렬
# authors 테이블에서 저자의 lastname, firstname 순으로 오름차순 정렬 후 출력
select *
from authors a
order by a.lastname, a.firstname asc;

# 문제 5. 책 제목에 ”How to Program＂을 포함하는 책 정보 출력
# titles 테이블에서 title 필드에 “How	 to Program”을 포함하는 책의 정보 출력
# isbn, title, edition_number, copyright 출력
# title 필드의 오름차순으로 정렬
select *
from titles as t
where t.title like '%How to Program%'
order by t.title asc;

# 문제 6. 내부 조인 #1
# authors 테이블과 author_isbn 테이블을 내부 조인
# 조인 기준: author_id가 동일
# 출력 내용: firstname, lastname, isbn
# 정렬 기준: lastname, firstname 기준 오름 차순

select a.firstname, a.lastname, ai.isbn
from authors as a inner join author_isbn as ai
on a.author_id = ai.author_id
order by a.lastname, a.firstname asc;

# 문제 7. 내부 조인 #2
# author_isbn 테이블과 titles 테이블을 내부 조인
# 조인 기준: isbn 동일
# 출력 내용: author_id, isbn, title, edition_number, copyright
# 정렬 기준: isbn 내림차순

select ai.author_id, t.isbn, t.title, t.edition_number, t.copyright
from author_isbn as ai inner join titles as t
on ai.isbn = t.isbn
order by ai.isbn desc;

# 문제 8. 3개의 테이블을 내부 조인
# lastname이 ‘Quirk’인 사람이 쓴 책 정보 출력
# 출력 내용: firstname, lastname, title, isbn, copyright

select a.firstname, a.lastname, t.title, ai.isbn, t.copyright
from titles as t
inner join author_isbn as ai on t.isbn = ai.isbn
inner join authors as a on a.author_id = ai.author_id
where a.lastname = 'Quirk';
	
# 문제 9. 3개의 테이블을 내부 조인
# ‘Paul	Deitel’ 또는 ‘Harvel Deitel’이 쓴 책 정보 출력
# 출력 내용: firstname, lastname, title, isbn, copyright

select a.firstname, a.lastname, t.title, t.isbn, t.copyright
from authors as a
	inner join author_isbn as ai on a.author_id = ai.author_id
	inner join titles as t on t.isbn = ai.isbn
where a.firstname in ('Paul','Harvey');


# 문제 10. 3개의 테이블을 내부 조인
# ‘Abbey Deitel’과 ‘Harvey Deitel’이 공동 저자인 책 정보 출력
# 출력 내용: firstname, lastname, title, isbn, copyright

select a.firstname, a.lastname, t.title, t.isbn, t.copyright
from author_isbn as ai
	inner join titles as t 
	on t.isbn = ai.isbn
	inner join authors as a
	on a.author_id = ai.author_id
	where a.firstname = 'Abbey';

select a.firstname, a.lastname, t.title, t.isbn, t.copyright
from author_isbn as ai
	inner join titles as t 
	on t.isbn = ai.isbn
	inner join authors as a
	on a.author_id = ai.author_id
	where a.firstname = 'Harvey';

select Abb.title, Abb.isbn, Abb.copyright
from
(select a.firstname, a.lastname, t.title, t.isbn, t.copyright
from author_isbn as ai
	inner join titles as t 
	on t.isbn = ai.isbn
	inner join authors as a
	on a.author_id = ai.author_id
	where a.firstname = 'Abbey') as Abb
inner join
(select a.firstname, a.lastname, t.title, t.isbn, t.copyright
from author_isbn as ai
	inner join titles as t 
	on t.isbn = ai.isbn
	inner join authors as a
	on a.author_id = ai.author_id
	where a.firstname = 'Harvey') as Harv
on Harv.title = Abb.title;