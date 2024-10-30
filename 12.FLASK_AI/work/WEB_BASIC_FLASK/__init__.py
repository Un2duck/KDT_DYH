# --------------------------------------------------------------------------------
# Flask Framework에서 WebServer 구동 파일
# - 파일명 : app.py
# --------------------------------------------------------------------------------

# 모듈 로딩
from flask import Flask, render_template

# 사용자 정의 함수

def create_app():

    # 전역 변수
    # Flask Web Server 인스턴스 생성
    APP = Flask(__name__)

    # 라우팅 기능 함수
    # Flask Web Server 인스턴스변수명.route("URL")

    # http://127.0.0.1:5000/main
    # http://127.0.0.1:5000/main/
    @APP.route('/main')
    @APP.route('/main/')
    def index():
        return render_template("hunmain.html")
    
    # 머신러닝 화면
    @APP.route("/main/ML")
    def ML():
        return render_template("ML.html")
    
    # 자연어처리 화면
    @APP.route("/main/NLP")
    def NLP():
        return render_template("NLP.html")

    # OpenCV 화면
    @APP.route("/main/CV")
    def CV():
        return render_template("OpenCV.html")

    # torchDL 화면
    @APP.route("/main/DL")
    def DL():
        return render_template("torchDL.html")
    
    return APP

# 조건부 실행
if __name__ == '__main__':
    # Flask Web Server 구동
    app = create_app()
    app.run()