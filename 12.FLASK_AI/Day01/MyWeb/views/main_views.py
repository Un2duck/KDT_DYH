# --------------------------------------------------------------------------------
# Flask Framework에서 '/'URL에 대한 라우팅 처리 파일
# - 파일명 : main_views.py
# --------------------------------------------------------------------------------
# 모듈 로딩
from flask import Blueprint, render_template

# Blueprint 인스턴스 생성
# http://127.0.0.1:5000/
main_bp = Blueprint('main',
                    __name__,
                    url_prefix='/',
                    template_folder='templates')

## 라우팅 기능 함수 정의
@main_bp.route('/', endpoint='hello') # flask에서의 endpoint : 처리함수의 별칭
def index():
    return render_template('index.html')