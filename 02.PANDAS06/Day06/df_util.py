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