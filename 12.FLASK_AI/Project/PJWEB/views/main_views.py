# --------------------------------------------------------------------------------
# Flask Framework에서 모듈단위 URL 처리 파일
# - 파일명 : main_views.py
# --------------------------------------------------------------------------------
# 모듈로딩

# from flask import Blueprint, render_template, url_for
from flask import Blueprint, render_template

# Blueprint 인스턴스 생성
BP = Blueprint('MAIN',
               import_name=__name__,
               url_prefix='/',
               template_folder='templates')

@BP.route('/')
def index():
    return render_template('pjindex.html', name='홍길동')