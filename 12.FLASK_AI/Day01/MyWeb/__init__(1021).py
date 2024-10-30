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

    # http://127.0.0.1:5000/info
    # http://127.0.0.1:5000/info/
    @APP.route('/')
    def index():
        return render_template("index.html")

    @APP.route("/info")
    @APP.route("/info/")
    def info():
        return """
        <body style='background-color:beige; text-align:center'>
            <h1>Hello~ Flask W E B~</h1>
        </body>
        """

    # http://127.0.0.1:5000/info/문자열 변수
    # name에 문자열 변수 저장 즉, name=문자열 변수
    @APP.route("/info/<name>")
    def printInfo(name):
        # return f"""
        # <body style='background-color:gray; text-align:center'>
        #     <h1>{name}'s INFOMATION </h1>
        # </body>
        # """
        return render_template("info.html", name=name)

    # http://127.0.0.1:5000/info/정수
    # age라는 변수에 정수 저장
    @APP.route("/info/<int:age>")
    def checkAge(age):
        return f"""
        <body style='background-color:skyblue; text-align:center'>
            <h1> 나이 : {age}살 입니다. </h1>
        </body>
        """

    # http://127.0.0.1:5000/go
    @APP.route("/go")
    def goHome():
        return APP.redirect('/info')
    
    return APP

# 조건부 실행
if __name__ == '__main__':
    # Flask Web Server 구동
    app = create_app()
    app.run()