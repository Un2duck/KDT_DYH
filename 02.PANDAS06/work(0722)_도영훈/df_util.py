# -------------------------------------------------------------------------
# Series/DataFrame에서 사용되는 사용자 정의 함수들
# -------------------------------------------------------------------------
# 함수기능 : DataFrame의 기본정보와 속성 확인 기능
# 함수이름 : checkDataFrame
# 매개변수 : DataFrame 인스턴스 변수명, DataFrame 인스턴스 이름
# 리턴값/반환값 : 없음
# -------------------------------------------------------------------------
def checkDataFrame(df_instance, name):
    print(f'\n[{name}]')
    df_instance.info()
    print(f'index: {df_instance.index}')
    print(f'columns: {df_instance.columns}')
    print(f'ndim: {df_instance.ndim}')
    print(f'shape: {df_instance.shape}')


def set_costomfont():
    # 한글폰트 설정 => 폰트 매니저 모듈
    from matplotlib import font_manager as fm, rc

    # 폰트 패밀리 이름 가져오기
    FONT_FILE = r"C:\Windows\Fonts\NANUMGOTHIC.ttf"
    font_name=fm.FontProperties(fname=FONT_FILE).get_name()

    # 새로운 폰트 패밀리 이름 지정
    rc('font', family=font_name)