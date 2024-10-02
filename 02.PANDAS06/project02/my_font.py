# 한글폰트 함수로 만들어두기

def set_costomfont():
    # 한글폰트 설정 => 폰트 매니저 모듈
    from matplotlib import font_manager as fm, rc

    # 폰트 패밀리 이름 가져오기
    FONT_FILE = r"C:\Windows\Fonts\NANUMGOTHIC.ttf"
    font_name=fm.FontProperties(fname=FONT_FILE).get_name()

    # 새로운 폰트 패밀리 이름 지정
    rc('font', family=font_name)