# 문자열 조작하기

use sqlclass_db;

select *
from customer;

-- drop table if exists string_tbl;
create table string_tbl
(char_fld CHAR(30),
vchar_fld VARCHAR(30),
text_fld TEXT
);

insert into string_tbl(char_fld, vchar_fld, text_fld)
values ('This is char data',
		'This is varhchar data',
		'This is text data');
		
select * from string_tbl;

update string_tbl
set vchar_fld = 'This is a piece of extremely long varchar data';

# 현재 모드 확인
select @@session.sql_mode;

#ansi 모드 선택
set sql_mode = 'ansi';

select vchar_fld from string_tbl;

delete from string_tbl;

insert into string_tbl(char_fld, vchar_fld, text_fld)
values ('This string is 28 characters',
		'This string is 28 characters',
		'This string is 28 characters');

# length() 함수: 문자열의 개수를 반환
select length(char_fld) as char_length,
length(vchar_fld) as varchar_length,
length(text_fld) as text_length
from string_tbl;

# position() 함수: 부분 문자열의 위치를 반환
select position('characters' in vchar_fld)
from string_tbl;

# locate('문자열', 열이름, 시작 위치) 함수
select locate('is', vchar_fld, 5)
from string_tbl;

select locate('is', vchar_fld, 1)
from string_tbl;

delete from string_tbl;

insert into string_tbl(vchar_fld)
values ('abcd'),
		('xyz'),
		('QRSTUV'),
		('qrstuv'),
		('12345');

select vchar_fld from string_tbl order by vchar_fld;

select strcmp('12345', '12345') 12345_12345,
strcmp('abcd', 'xyz') abcd_xyz,
strcmp('abcd', 'QRSTUV') abcd_QRSTUV,
strcmp('qrstuv', 'QRSTUV') qrstuv_QRSTUV,
strcmp('12345', 'xyz') 12345_xyz,
strcmp('xyz', 'qrstuv') xyz_qrstuv;

use sakila;

# like 또는 regexp 연산자
select name, name like '%y' as ends_in_y
from category;

select name, name regexp 'y$' as ends_in_y
from category;


use sqlclass_db;
delete from string_tbl;

insert into string_tbl (text_fld)
values ('This string was 29 characters');

select text_fld
from string_tbl;

# concat(): 문자열 추가
update string_tbl
set text_fld = concat(text_fld, ', but now it is longer');

use sakila;
# concat() 함수 사용 #2
select concat(first_name, ' ', last_name, ' has been a customer since ', date(create_date))
as cust_narrative
from customer;

# insert() 함수 : insert(문자열, 시작위치, 길이, 새로운 문자열)
select insert('goodbye world', 9, 0, 'cruel ') as string;
select insert('goodbye world', 9, 7, 'hello ') as string;

# replace() 함수 : replace(문자열, 기존문자열, 새로운 문자열)
select replace('goodbye world', 'goodbye', 'hello') as replace_str;

# substr() : substr(문자열, 시작위치, 개수) 또는 substring() 함수 
select substr('goodbye cruel world', 9, 5);

# 산술 함수
# round() : 반올림, truncate() : 버림
select round(72.0909, 1), round(72.0909, 2), round(72.0909, 3);
select truncate(72.0956, 1), truncate(72.0956, 2), truncate(72.0956, 3);

# 시간데이터 처리
# cast() 함수
select cast('2019-09-17 15:30:00' as datetime);

select cast('2019-09-17' as date) date_field,
cast('108:17:57' as time) time_field;

select cast('20190917153000' as datetime);

# 날짜 생성 함수 str_to_date(str, format)

select str_to_date('September 17, 2019', '%M %d, %Y') as return_date;
select str_to_date('04/30/2024', '%m/%d/%Y') as date1;

select str_to_date('01,5,2024', '%d,%m,%Y') as date2;

select str_to_date('15:35:00', '%H:%i:%s') as time1;

select datediff('2024-06-11', '2024-08-06');