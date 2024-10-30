# --------------------------------------------------------------------------------
# Flask Framework에서 WebServer 구동 파일
# - 파일명 : __init__.py
# --------------------------------------------------------------------------------

# 모듈 로딩
from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

# DB 관련 설정
import pjconfig

DB = SQLAlchemy()
MIGRATE = Migrate()

# --------------------------------------------------------------------------------
# Application 생성 함수
# - 함수명 : create_app <= 이름 변경 불가!!
# 사용자 정의 함수
# --------------------------------------------------------------------------------

def create_app():
    # 전역 변수
    # Flask Web Server 인스턴스 생성
    APP = Flask(__name__)

    # DB 관련 초기화 설정
    APP.config.from_object(pjconfig)

    # DB 초기화 및 연동
    DB.init_app(APP)
    MIGRATE.init_app(APP, DB)

    # # DB 클래스 정의 모듈
    # from .models import models

    # URL 처리 모듈 등록
    from .views import main_views
    APP.register_blueprint(main_views.BP)

    return APP

