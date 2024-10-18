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
import sys, codecs                      # 인코딩 관련
from pydoc import html                  # html 코드 관련 : html을 객체로 처리?
import pickle

# ---------------------------------------------------------------------
# torch 관련 모듈 로딩
# ---------------------------------------------------------------------

import torch

# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True                      # Jupyter Mode : False, WEB Mode : True
cgitb.enable()                          # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능

# 모델 호출
MODEL_FILE = r'C:\Users\KDP-50\OneDrive\바탕 화면\KDT_DYH\11.HTML,CSS,JAVA\Project\models\NLP_score(0.8598)_loss(0.3825).pth'
NLP_Model = torch.load(MODEL_FILE, weights_only=False)

MAX_LENGTH=50

# 저장된 vocab 파일 불러오기
with open('hun_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


# 사용자 정의 함수---------------------------------------------------------
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
          <title>---질병 여부 판별---</title>
         </head>
         <body bgcolor="antiquewhite">
          <form>
            <textarea name="text" rows="10" colos="40" >
                {text}
            </textarea>
            <p>
                <input type="submit" value="질병 판별!!">
                {msg}
            </p>
          </form>
         </body>
        </html>""")
    
# ---------------------------------------------------------------------
# 함수 이름 : predict_model()
# 함수 역할 : 모델 예측 함수
# 매개 변수 : model, img
# ---------------------------------------------------------------------

def predict_model(model, data, vocab, max_length):
    try:
        # 데이터가 문자열이라면 토큰화 및 인덱스로 변환
        tokens = [vocab.get(token, vocab['oov']) for token in data]  # 토큰을 인덱스로 변환

        # 패딩하여 모델의 입력 형태와 일치시키기
        if len(tokens) < max_length:
            tokens = tokens + [vocab['pad']] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]

        dataTS = torch.LongTensor(tokens).unsqueeze(0)

        # 검증 모드로 모델 설정
        model.eval()
        with torch.no_grad():
            # 추론/평가
            logits = model(dataTS)
            pre_val = torch.sigmoid(logits)

        prediction = (pre_val >= 0.5).float()
        check = '해당 질병 없습니다.' if prediction.item() else '해당 질병 있습니다.'
        return check
    
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

# (3) WEB 사용자 입력 데이터 처리
# (3-1) HTML 코드에서 사용자 입력 받는 form 태그 영역 객체 가져오기
form = cgi.FieldStorage()

# (3-2) Form안에 textarea 태그 속 데이터 가져오기
text = form.getvalue("text", default="")

# (3-3) 판별하기
msg = ""
if text != "":
    result = predict_model(NLP_Model, text, vocab, MAX_LENGTH)
    msg = f"{result}"

# (4) 사용자에게 WEB 화면 제공
showHTML(text, msg)