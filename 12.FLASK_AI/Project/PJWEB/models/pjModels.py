# --------------------------------------------------------------------------------
# 데이터베이스의 테이블 정의 클래스
# --------------------------------------------------------------------------------
# 모듈 로딩
from PJWEB import DB

# --------------------------------------------------------------------------------
# Features 테이블 정의 클래스
# - PK : id
# --------------------------------------------------------------------------------
class Features(DB.Model):
    id = DB.Column(DB.Integer, primary_key=True)
    subject = DB.Column(DB.String(200), nullable=False)
    content = DB.Column(DB.Text(), nullable=False)
    create_date = DB.Column(DB.DateTime(), nullable=False)

# --------------------------------------------------------------------------------
# Target 테이블 정의 클래스
# - PK : id
# - FK : Features.id
# --------------------------------------------------------------------------------
class Target(DB.Model):
    id = DB.Column(DB.Integer, primary_key=True)
    Features_id = DB.Column(DB.Integer,
                            DB.ForeignKey('Features_id'))