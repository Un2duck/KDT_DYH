# --------------------------------------------------------------------
# Flask Framework에서 WebServer 구동 파일
# - 파일명 : __init__.py
# --------------------------------------------------------------------
# 모듈 로딩 -----------------------------------------------------------
from models.NLP_Class import *
from flask import Flask, render_template, request, redirect, url_for

# 전역변수 ------------------------------------------------------------
# Flask Web Server 인스턴스 생성
APP=Flask(__name__)

# 라우팅 기능 함수 ----------------------------------------------------
# Flask Web Server 인스턴스 변수명.route("URL")
# http://127.0.0.1:5000/
@APP.route('/')
def login():
    return render_template('login.html')

# http://127.0.0.1:5000/customer_mainpage
@APP.route("/customer_mainpage", methods=["GET", "POST"])
def customer_mainpage():
    result_list = None
    if request.method == "POST":

        # 텍스트 데이터 가져오기
        message = request.form.get("message")
        
        # LSTM 모델 경로
        LSTM_model_path = r'C:\Users\KDP-50\OneDrive\바탕 화면\KDT_DYH\14.personal_project(final)2\nlpWEB\models\LSTM_9_loss(0.05198).pth'

        # 모델 예측 결과 가져오기
        result_list = predict_LSTM(LSTM_model_path, [message])
    
    # output = predict_LSTM(LSTM_model_path, sample_texts)

    return render_template("customer_mainpage.html", result_list = result_list)

# 조건부 실행 ---------------------------------------------------------
if __name__ == '__main__':
    
    APP.run()