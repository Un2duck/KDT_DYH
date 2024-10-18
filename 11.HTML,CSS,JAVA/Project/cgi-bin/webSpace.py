# 위에 라인 : 셀 내용을 파일로 생성/ 한번 생성후에는 마스킹

# --------------------------------------------------------------------------
# 경로 지정
# --------------------------------------------------------------------------
import sys
sys.path.append(r'C:\Users\KDP-50\OneDrive\바탕 화면\KDT_DYH\11.HTML,CSS,JAVA\Project\MyClass')

# --------------------------------------------------------------------------
# 모듈 로딩
# --------------------------------------------------------------------------

import os.path                          # 파일 및 폴더 관련
import cgi, cgitb                       # cgi 프로그래밍 관련
import joblib                           # AI 모델 관련
import sys, codecs                      # 인코딩 관련
from pydoc import html                  # html 코드 관련 : html을 객체로 처리?
import pickle

import pandas as pd
import numpy as np

# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True                      # Jupyter Mode : False, WEB Mode : True
cgitb.enable()                          # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능

# 모델 경로
MODEL_FILE = r'C:\Users\KDP-50\OneDrive\바탕 화면\KDT_DYH\11.HTML,CSS,JAVA\Project\models\bestspace.pth'
# MODEL_FILE = 'weather_loss(2.01475)_score(0.73614).pth'

# 모델 호출

with open(MODEL_FILE, 'rb') as f:
    space_best_model = pickle.load(f)

# 사용자 정의 함수----------------------------------------------------------
# WEB에서 사용자에게 보여주고 입력받는 함수 ---------------------------------
# 함수명 : showHTML
# 재 료 : 사용자 입력 데이터, 판별 결과
# 결 과 : 사용자에게 보여질 HTML 코드

def showHTML(text, msg):
    print("Content-Type: text/html; charset=utf-8")
    print(f"""
    
        <!DOCTYPE html>
        <html lang="en">
         <head>
          <meta charset="UTF-8">
          <title>---우주 생존 예측!---</title>
         </head>
         <body bgcolor="gray">
          <form>
            <textarea name="text" rows="10" colos="40" >
                {text}
            </textarea>
            <p>
                <input type="submit" value="우주 생존 예측!!">
                    {msg}
            </p>
          </form>
         </body>
        </html>""")

# ---------------------------------------------------------------------
# 함수 이름 : predice_model()
# 함수 역할 : 모델 예측 함수
# 매개 변수 : model, data
# ---------------------------------------------------------------------

def predict_model(model, data):
    data = [float(x) for x in data.split(',')]
    data = pd.Series(data)
    data = data.to_numpy().reshape(1, -1)
    result = model.predict(data)
    result = '사망...!' if result == 0 else '생존...!'
    try:
        return result
    
    except Exception as e:
        print(f"Error during prediction: {e}") # 에러 로그 추가
        return '오류 발생!!'

# --------------------------------------------------------------------------
# 기능 구현 
# --------------------------------------------------------------------------
# (1) WEB 인코딩 설정
if SCRIPT_MODE:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()) # 웹에서만 필요 : 표준출력을 utf-8로

# (2) 모델 로딩
if SCRIPT_MODE:
    pthfile = os.path.dirname(__file__)+ MODEL_FILE # 웹상에서는 절대경로만
else:
    pthfile = MODEL_FILE
    
# langModel = joblib.load(pthfile)

# (3) WEB 사용자 입력 데이터 처리
# (3-1) HTML 코드에서 사용자 입력 받는 form 태그 영역 객체 가져오기
form = cgi.FieldStorage()

# (3-2) Form안에 textarea 태그 속 데이터 가져오기
text = form.getvalue("text", default="")

# (3-3) 판별하기
msg = ""
if text != "":
    result = predict_model(space_best_model, text)
    msg = f"{result}"

# (4) 사용자에게 WEB 화면 제공
showHTML(text, msg)