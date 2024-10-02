# sqlclass_db 사용하기
use sqlclass_db;

# nobel 테이블 확인하기
select * from nobel;

# 1) 1960년에 노벨상 물리학상 또는 노벨 평화상을 수상한 사람의 정보를 출력
# - 출력 컬럼: year, category, fullname
select n.year, n.category, n.fullname
from nobel as n
where (n.year = 1960 and n.category = 'physics') or (n.year = 1960 and n.category = 'Peace');

# 2) Albert	Einstein이 수상한 연도와 수상 분야(category), 출생대륙, 출생국가를 출력
# - 출력 컬럼: year, category, amount, birth_continent, birth_country
select n.year, n.category, n.prize_amount, n.birth_continent, n.birth_country
from nobel as n
where n.fullname = 'Albert Einstein';

# 3) 1910년부터 2010년까지 노벨 평화상 수상자 명단 출력 (연도 오름차순)
# - 출력 컬럼: year, fullname, birth_country
select n.year, n.fullname, n.birth_country
from nobel as n
where n.year >= 1910 and n.year <= 2010;

# 4) 전체 테이블에서 수상자 이름이 ‘John’으로 시작하는 수상자 모두 출력
# - 출력 컬럼: category, fullname
select n.category, n.fullname
from nobel as n
where n.fullname like 'John%';

# 5) 1964년 수상자 중에서 노벨화학상과 의학상(‘Physiology	or	Medicine’)을 제외한
# 수상자의 모든 정보를 수상자 이름을 기준으로 오름차순으로 정렬 후 출력
select *
from nobel as n
where (n.year = 1964) and (n.category <> 'Physiology or Medicine');

# 6) 2000년부터 2019년까지 노벨 문학상 수상자 명단 출력
# - 출력 컬럼: year, fullname, gender, birth_country
select n.year, n.fullname, n.gender, n.birth_country
from nobel as n
where (n.year >= 2000) and (n.year <= 2019) and (n.category = 'Literature');

# 7) 각 분야별 역대 수상자의 수를 내림차순으로 정렬 후 출력(group by, order by)
select n.category, count(*) as prize_count
from nobel as n
group by n.category
order by count(*) desc;

# 8) 노벨 의학상이 있었던 연도를 모두 출력 (distinct)	사용
select distinct n.year
from nobel as n
where n.category = 'Physiology or Medicine';

# 9) 노벨 의학상이 없었던 연도의 총 회수를 출력
select n.category, count(*) as prize_count
from nobel as n
where n.category <> 'Physiology or Medicine'
group by n.category;
order by count(*) desc;

# 10) 여성 노벨 수상자의 명단 출력
# - 출력 컬럼: fullname, category, birth_country
select n.fullname, n.category, n.birth_country
from nobel as n
where n.gender = 'female';

# 11) 수상자들의 출생 국가별 횟수 출력
# - 출력 컬럼: birth_country, 횟수
select n.birth_country, count(*) as prize_count
from nobel as n
group by n.birth_country
order by count(*) desc;

# 12) 수상자의 출생 국가가 ‘Korea’인 정보 모두 출력
select *
from nobel as n
where n.birth_country = 'Korea';

# 13) 수상자의 출신 국가가 (‘Europe’,	‘North America’, 공백)이 아닌 모든 정보 출력
select *
from nobel as n
where (n.birth_country <> 'Europe') and (n.birth_country <> 'North America') and (n.birth_country <> '');

# 14) 수상자의 출신 국가별로 그룹화를 해서 수상 횟수가 10회 이상인 국가의 모든 정보
# 출력하고 국가별 수상횟수의 역순으로 출력 (birth_country, 횟수 출력)
select n.birth_country, count(*) as prize_count
from nobel as n
group by n.birth_country
having count(*) >= 10
order by count(*) desc;

# 15) 2회 이상 수상자 중에서 fullname이 공백이 아닌 경우를 출력하는데, fullname의 오름차순으로 출력
# - 출력 컬럼: fullname,	횟수
select n.fullname, count(*) as prize_count
from nobel as n
group by n.fullname
having count(*) <= 2
order by count(*) desc;