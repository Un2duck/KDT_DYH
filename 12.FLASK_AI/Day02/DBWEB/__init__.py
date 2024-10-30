# --------------------------------------------------------------------------------
# Flask Framework에서 WebServer 구동 파일
# - 파일명 : __init__.py
# --------------------------------------------------------------------------------

# 모듈 로딩
from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

# DB관련 설정
import config

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
    APP.config.from_object(config)
    
    # DB 초기화 및 연동
    DB.init_app(APP)
    MIGRATE.init_app(APP, DB)

    # DB 클래스 정의 모듈
    from .models import models

    # # URL 즉, 클라이언트 요청 페이지 주소를 보여줄 기능 함수
    # def printPage():
    #     return "<h1>hello~ </h1>"
    # # URL처럼 함수 연결
    # APP.add_url_rule('/', view_func=printPage, endpoint='INDEX')

    # URL 즉, 클라이언트 요청 페이지 주소를 보여줄 기능 함수
    # @APP.route('/')
    # def printPage():
    #     return "<h1>hello~ </h1>"

    # URL 처리 모듈 등록
    from .views import main_view
    APP.register_blueprint(main_view.mainBP)



    return APP