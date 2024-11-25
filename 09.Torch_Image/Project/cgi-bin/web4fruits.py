# ---------------------------------------------------------------------
# 경로 지정
# ---------------------------------------------------------------------
import sys
sys.path.append(r'C:\Users\KDP-50\OneDrive\바탕 화면\KDT_DYH\MyClass')

# ---------------------------------------------------------------------
# 모듈 로딩
# ---------------------------------------------------------------------

from DL_Modules import *
from DL_func import *
from DL_Reg_Class import *
from ML_preprocessing import *

import os.path                          # 파일 및 폴더 관련
import cgi, cgitb                       # cgi 프로그래밍 관련
import joblib                           # AI 모델 관련
import sys, codecs                      # 인코딩 관련
from pydoc import html                  # html 코드 관련 : html을 객체로 처리?

# ---------------------------------------------------------------------
# 이미지 관련 모듈 로딩
# ---------------------------------------------------------------------

import cv2
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

# ---------------------------------------------------------------------
# 동작관련 전역 변수 ----------------------------------------------------
# ---------------------------------------------------------------------
SCRIPT_MODE = True                      # Jupyter Mode : False, WEB Mode : True
cgitb.enable()                          # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능

# 모델 호출
# MODEL_FILE = 'loss(2.01475)_score(0.73614).pth'
MODEL_FILE = r'C:\Users\KDP-50\OneDrive\바탕 화면\KDT_DYH\09.Torch_Image\Project\models\model_num_loss(8.6757)_score(0.9410)'
fruits_Model = torch.load(MODEL_FILE, weights_only=False)

## 데이터 변형 및 전처리
transConvert = v2.Compose([
    v2.Resize([256, 256]),
    v2.RandomResizedCrop(224),
    v2.ToTensor(),
    # v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.ToDtype(torch.float32, scale=True)
])

# 과일 이름 딕셔너리
check_fruit_dict = {'apple':0,'banana':1,'orange':2,'strawberry':3}

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
          <title>--4fruits(apple, banana, orange, strawberry) predict --</title>
         </head>
         <body bgcolor="pink">
            <form method="POST"  enctype="multipart/form-data">
                <div>
                    file : <input type="file" id="imageInput" name="file1" accept="image/*">
                </div>
                <input type="submit" value="4가지 과일 예측해보기!">{msg}</p>
          </form>
         </body>
        </html>""")

# ---------------------------------------------------------------------
# 함수 이름 : predict_model()
# 함수 역할 : 모델 예측 함수
# 매개 변수 : model, img
# ---------------------------------------------------------------------

def predict_model(model, img):
    try:
        print("img received for prediction:", img) # 디버깅용 로그'
        img = Image.open(img)
        img = transConvert(img)
        img = img.unsqueeze(0)
        # img = img.to(DEVICE)

        # 검증 모드로 모델 설정
        model.eval()
        with torch.no_grad():

            # 추론/평가
            pre_val = model(img)
            pre_val=pre_val.argmax().item()
            result = [key for key, val in check_fruit_dict.items() if val == pre_val][0]
        return f'해당 이미지는 {result} 입니다'
    
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

# (3-2) 이미지 데이터 가져오기
# if form.getvalue("file", None):
image = form.getvalue("file1", None)

if image: 
    filename = form['file1'].filename

# (3-3) 판별하기
# if image and "file" in form and form["file"].filename:

if image:
    result = predict_model(fruits_Model, f'images\{filename}')
    msg = f"예측 결과: {result}"
    print(msg)
else:
    msg = '이미지가 없습니다. (이미지를 첨부하세요.!)'
    print(msg)

# (4) 사용자에게 WEB 화면 제공
showHTML("", msg)