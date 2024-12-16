# --------------------------------------------------------------------
# Flask Framework에서 WebServer 구동 파일
# - 파일명 : app.py 
# --------------------------------------------------------------------
# 모듈 로딩 -----------------------------------------------------------
from models.ProjectModule import *
from flask import Flask, render_template, request, redirect, url_for
import pymysql
from datetime import datetime, timedelta
import calendar
import os
import numpy as np
from flask import flash
from collections import defaultdict

# 전역변수 ------------------------------------------------------------
# Flask Web Server 인스턴스 생성
APP=Flask(__name__)
APP.secret_key = "1234"

# DB 연결 함수
def get_db_connection():
    return pymysql.connect(
        host='172.20.146.27',
        user='younghun',
        password='1234',
        db='quasar_copy',
        charset='utf8'
    )

conn = get_db_connection()

# 라우팅 기능 함수 ----------------------------------------------------
# Flask Web Server 인스턴스 변수명.route("URL")
# http://127.0.0.1:5000/
@APP.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('user_id')  # 사용자 입력 아이디
        password = request.form.get('password')  # 사용자 입력 비밀번호
        user_type = request.form.get('role')  # 사용자 유형 (개인/기관)

        if not user_type:  # 사용자 유형 선택 여부 확인
            return render_template('login.html', message="사용자 유형을 선택해주세요.")
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 사용자 유형에 따른 쿼리
            if user_type == '개인':  # 개인
                query = "SELECT Member_Password FROM Member WHERE Member_ID = %s"
            elif user_type == '기관':  # 기관
                query = "SELECT Password FROM Office WHERE Office_ID = %s"
            else:
                return render_template('login.html', message="사용자 유형을 선택해주세요.")

            # 쿼리 실행
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()

            if result:
                stored_password = result[0]
                if stored_password == password:  # 비밀번호 일치
                    if user_type == '개인':
                        return redirect(url_for('customer_mainpage'))
                    elif user_type == '기관':
                        return redirect(url_for('admin_mainpage'))
                else:
                    return render_template('login.html', message="ID/비밀번호가 잘못되었습니다.")
            else:
                return render_template('login.html', message="ID/비밀번호가 잘못되었습니다.")

        except pymysql.MySQLError as e:
            print(f"Database error: {e}")
            return render_template('login.html', message="데이터베이스 오류가 발생했습니다.")

    return render_template('login.html')

# http://127.0.0.1:5000/find_id
@APP.route('/find_id', methods=['GET','POST'])
def find_id():
    message = None # 기본값 설정
    if request.method == 'POST':
        # 폼 데이터 수집
        resident1 = request.form.get('resident1')
        resident2 = request.form.get('resident2')
        phone_prefix = request.form.get('phone_prefix')
        phone2 = request.form.get('phone2')
        phone3 = request.form.get('phone3')

        # 주민등록번호 및 전화번호 조합
        resident_number = f"{resident1}-{resident2}"
        phone_number = f"{phone_prefix}-{phone2}-{phone3}"

        # DB 검색
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = """
                SELECT member_id FROM member
                WHERE registration_number = %s AND member_phone_number = %s
            """
            cursor.execute(query, (resident_number, phone_number))
            result = cursor.fetchone() # 결과 가져오기

            cursor.close()
            conn.close()

            if result:
                # 아이디를 찾은 경우
                message=f"아이디는 '{result[0]}'입니다."                
            else:
                # 아이디를 찾지 못한 경우
                message="입력된 정보와 일치하는 아이디가 없습니다. 다시 입력해주세요."

        except pymysql.MySQLError as e:
            print(f"Database error: {e}")
    return render_template('/find_id.html', message=message)

# http://127.0.0.1:5000/find_pw
@APP.route('/find_pw', methods=['GET','POST'])
def find_pw():
    message = None # 기본값 설정
    if request.method == 'POST':
        # 폼 데이터 수집
        member_id = request.form.get('username')
        resident1 = request.form.get('resident1')
        resident2 = request.form.get('resident2')
        phone_prefix = request.form.get('phone_prefix')
        phone2 = request.form.get('phone2')
        phone3 = request.form.get('phone3')

        # 주민등록번호 및 전화번호 조합
        member = f"{member_id}"
        resident_number = f"{resident1}-{resident2}"
        phone_number = f"{phone_prefix}-{phone2}-{phone3}"

        # DB 검색
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = """
                SELECT member_password FROM member
                WHERE member_id = %s AND registration_number = %s AND member_phone_number = %s
            """
            cursor.execute(query, (member, resident_number, phone_number))
            result = cursor.fetchone() # 결과 가져오기

            cursor.close()
            conn.close()

            if result:
                # 비밀번호를 찾은 경우
                message=f"비밀번호는 '{result[0]}'입니다."                
            else:
                # 비밀번호를 찾지 못한 경우
                message="잘못된 정보입니다. 다시 입력해주세요."

        except pymysql.MySQLError as e:
            print(f"Database error: {e}")
    return render_template('/find_pw.html', message=message)

# http://127.0.0.1:5000/customer_gohome
@APP.route("/customer_gohome")
def customer_gohome():
    return APP.redirect("/customer_mainpage.html")

# http://127.0.0.1:5000/admin_gohome
@APP.route("/admin_gohome")
def admin_gohome():
    return APP.redirect("/admin_mainpage.html")

# http://127.0.0.1:5000/admin_mainpage
@APP.route("/admin_mainpage", methods=["GET", "POST"])
def admin_mainpage():

    '''
    # DB 연결
    '''

    # 거주자 정보 가져오기
    cur = conn.cursor()
    query = "SELECT dong, ho, water_condition FROM managed_entity"
    cur.execute(query)
    status = cur.fetchall()
    cur.close()
    # print("Fetched status data:", status) # 데이터 확인
    
    # 팝업창 띄울 샘플만들기
    cur2 = conn.cursor()
    query2 = """
        SELECT
            ho,
            managed_entity_name,
            phone_number,
            water_condition
        FROM 
            managed_entity;
        """
    cur2.execute(query2)
    popup = cur2.fetchall()
    cur2.close()

    # 오른쪽 박스에 요약 나타내기
    cur3 = conn.cursor()
    query3 = """ 
        SELECT 
            water_condition, COUNT(*) AS count
        FROM managed_entity
        WHERE dong = 101
        GROUP BY water_condition
        """
    cur3.execute(query3)
    rightbox = cur3.fetchall()
    cur3.close()

    # 위험 상태 정보 가져오기
    cur4 = conn.cursor()
    query4 = """
        SELECT dong, ho
        FROM managed_entity
        WHERE water_condition = 'danger'
        """
    cur4.execute(query4)
    danger_result = cur4.fetchall()
    cur4.close()

    # 공지사항에 최신 1개 데이터 가져오기
    cur5 = conn.cursor()
    query5 = "SELECT announcement_id, announcement_title, announcement_content, announcement_date FROM announcement ORDER BY announcement_id DESC LIMIT 1"
    cur5.execute(query5)
    announcement = cur5.fetchone()
    cur5.close()

    # 게시판에 최신 5개 데이터 가져오기
    cur6 = conn.cursor()
    query6 = "SELECT board_id, board_title, board_author, board_date FROM board ORDER BY board_id DESC LIMIT 5"
    cur6.execute(query6)
    board_data = cur6.fetchall()
    cur6.close()

    cur7 = conn.cursor()
    query7 = "SELECT member_id, water_usage FROM water ORDER BY member_id, water_date, no_id"
    cur7.execute(query7)
    water_data = cur7.fetchall()
    cur7.close()

    cur8 = conn.cursor()
    query8 = "SELECT member_id, electric_usage FROM electric ORDER BY member_id, electric_date, no_id"
    cur8.execute(query8)
    elec_data = cur8.fetchall()
    cur8.close()

    water_list = split_tuple(water_data, 10)
    elec_list = split_tuple(elec_data, 10)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    water_model_path = os.path.join(BASE_DIR, "models/water_autoencoder_model.pth")
    electric_model_path = os.path.join(BASE_DIR, "models/electric_autoencoder_model.pth")

    water_scaler_path = os.path.join(BASE_DIR, "models/water_mm_scaler.pkl")

    status_list = []

    total_elec_error = []
    total_water_error = []
    total_water = []
    total_elec = []
    total_predict_water = []
    total_predict_elec = []

    for i in range(len(water_list)):
        output1 = [row for row in water_list[i][-28:]]  # water_usage 변환
        output2 = [row for row in elec_list[i][-28:]]  # electric_usage 변환

        output1 = output1[-28:]
        output2 = output2[-28:]

        ElecErrorMargin = []

        for elec in output2:
            if elec <= 1.4:
                ElecErrorMargin.append(0.08)
            elif elec <= 1.6:
                ElecErrorMargin.append(0.13)
            elif elec <= 2:
                ElecErrorMargin.append(0.15)
        
        WaterErrorMargin = []

        for water in output1:
            WaterErrorMargin.append(40)

        water_ts, water_scaler = preprocessing(output1, water_scaler_path)
        elec_ts = torch.FloatTensor([output2])

        water_model = load_water_model(water_model_path)
        electric_model = load_electric_model(electric_model_path)

        predict_electric = electric_model(elec_ts).squeeze(0).tolist()
        predict_water = water_model(water_ts)

        predict_water_original = water_scaler.inverse_transform(predict_water.detach().numpy()).squeeze(0).tolist()

        cnt = 0

        for i in range(4):
            if not (predict_water_original[23 + i] - WaterErrorMargin[23 + i]) <= output1[23 + i] <= (predict_water_original[23 + i] + WaterErrorMargin[23 + i]):
                if not (predict_electric[23 + i] - ElecErrorMargin[23 + i]) <= output2[23 + i] <= (predict_electric[23 + i] + ElecErrorMargin[23 + i]):
                    cnt += 1

        if cnt >= 2:
            status_list.append('danger')
        elif cnt >= 1:
            status_list.append('caution')
        else:
            status_list.append('normal')

        s = set()
        for data in water_data:
            s.add(data[0])
            id_list = sorted(s)

        total_water_error.append(WaterErrorMargin)
        total_elec_error.append(ElecErrorMargin)
        total_water.append(output1)
        total_elec.append(elec_ts.squeeze(0).tolist())
        total_predict_water.append(predict_water_original)
        total_predict_elec.append(predict_electric)

    cur9 = conn.cursor()
    query9 = "SELECT member_id FROM managed_entity ORDER BY member_id LIMIT 10;"
    cur9.execute(query9)
    result = cur9.fetchall()

    # 2. 업데이트 실행
    ids = [row[0] for row in result]  # 추출된 ID 리스트

    for idx, new_value in zip(ids, status_list):
        sql_update = "UPDATE managed_entity SET water_condition = %s WHERE member_id = %s;"
        cur9.execute(sql_update, (new_value, idx))

    # 변경사항 커밋
    conn.commit()
    cur9.close()

    # 공지사항이 없으면 None을 전달하고, 있으면 공지사항 데이터를 전달
    if announcement:
        announcement_data = {
            'title': announcement[1],
            'content': announcement[2],
            'date': announcement[3],
            'idx' : announcement[0]
        }
    else:
        announcement_data = None
    
    # if board:
    #     board_data = {
    #         'title': board[1],
    #         'content': board[2],
    #         'date': board[3]
    #     }
    # else:
    #     board_data = None

    '''
    # 동 선택 부분
    '''

    # 기본적으로 101동이 선택됨
    selected_building = "101"
    if request.method == "POST":
        # POST 요청에서 선택된 동을 가져옴
        selected_building = request.form.get("building", "101")

    cur10 = conn.cursor()
    query10 = "SELECT water_condition, dong FROM managed_entity ORDER BY dong, ho"
    cur10.execute(query10)
    dong_status = cur10.fetchall()
    cur10.close()

    s = set()
    for dong in dong_status:
        s.add(dong[1])

    dong_list = sorted(s)

    dong_status_list = split_tuple2(dong_status, len(dong_list))

    building_status = {}

    normal = False
    caution = False
    danger = False

    for i in range(len(dong_status_list)):
        for j in range(len(dong_status_list[i])):
            if dong_status_list[i][j] == 'normal':
                normal = True
            elif dong_status_list[i][j] == 'caution':
                caution = True
            elif dong_status_list[i][j] == 'danger':
                danger = True
        if danger == True:
            building_status[f'{dong_list[i]}'] = 'danger'
        elif caution == True:
            building_status[f'{dong_list[i]}'] = 'caution'
        else:
            building_status[f'{dong_list[i]}'] = 'normal'
        
        normal = False
        caution = False
        danger = False

    dong_list = ['102', '103', '104', '105', '106', '107', '108', '109', '110']
    status_list = ['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'caution', 'normal']

    for i in range(len(dong_list)):
        building_status[f'{dong_list[i]}'] = f'{status_list[i]}'
    

    # 각 동별 데이터
    floors_data = {
        "101": {
            # "1-5": [["normal", "normal"], ["normal", "danger"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            # "6-10": [["normal", "normal"], ["danger", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            # "11-15": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "danger"], ["normal", "normal"]],
            # "16-20": [["normal", "normal"], ["caution", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            # "21-25": [["danger", "danger"], ["normal", "normal"], ["danger", "normal"], ["normal", "normal"], ["normal", "normal"]],
        },
        "102": {
            "1-5": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "6-10": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "11-15": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "16-20": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "21-25": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
        },
        "103": {
            "1-5": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "6-10": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "11-15": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "16-20": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "21-25": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
        },
        "104": {
            "1-5": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "6-10": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "11-15": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "16-20": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "21-25": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
        },
        "105": {
            "1-5": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "6-10": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "11-15": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "16-20": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "21-25": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
        },
        "106": {
            "1-5": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "6-10": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "11-15": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "16-20": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "21-25": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
        },
        "107": {
            "1-5": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "6-10": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "11-15": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "16-20": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "21-25": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
        },
        "108": {
            "1-5": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "6-10": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "11-15": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "16-20": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "21-25": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
        },
        "109": {
            "1-5": [["normal", "normal"], ["normal", "caution"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "6-10": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "11-15": [["normal", "normal"], ["caution", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "16-20": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "21-25": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
        },
        "110": {
            "1-5": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "6-10": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "11-15": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "16-20": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
            "21-25": [["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"], ["normal", "normal"]],
        }
    }

    # floors_ho는 고정된 값
    floors_ho = {
        "1-5": [["101", "102"], ["201", "202"], ["301", "302"], ["401", "402"], ["501", "502"]],
        "6-10": [["601", "602"], ["701", "702"], ["801", "802"], ["901", "902"], ["1001", "1002"]],
        "11-15": [["1101", "1102"], ["1201", "1202"], ["1301", "1302"], ["1401", "1402"], ["1501", "1502"]],
        "16-20": [["1601", "1602"], ["1701", "1702"], ["1801", "1802"], ["1901", "1902"], ["2001", "2002"]],
        "21-25": [["2101", "2102"], ["2201", "2202"], ["2301", "2302"], ["2401", "2402"], ["2501", "2502"]],
    }

    '''
    # 거주자 샘플 데이터
    '''

    # resident_data = {
    #     "2101": {
    #         "member_id": "user001",
    #         "member_address": "서울특별시 강남구 도곡동",
    #         "member_phone_number": "010-1234-5678",
    #     },
    #     "2102": {
    #         "member_id": "user002",
    #         "member_address": "부산광역시 해운대구 우동",
    #         "member_phone_number": "010-9876-5432",
    #     },
    # }

    # DB 사용
    resident_data = {
        str(row[0]): {  # ho 값을 키로 사용
            "managed_entity_name": row[1],
            "phone_number": row[2],
            "water_condition": row[3],
            "water_usage": total_water[idx % 10] if idx % 10 < len(total_water) else None,
            "elec_usage": total_elec[idx % 10] if idx % 10 < len(total_elec) else None,
            "water_predict_usage": total_predict_water[idx % 10] if idx % 10 < len(total_predict_water) else None,
            "elec_predict_usage": total_predict_elec[idx % 10] if idx % 10 < len(total_predict_elec) else None,
            "water_error": total_water_error[idx % 10] if idx % 10 < len(total_water_error) else None,
            "elec_error": total_elec_error[idx % 10] if idx % 10 < len(total_elec_error) else None
        }
        for idx, row in zip(range(len(popup)), popup)
    }
    
    # 선택된 동의 floors 데이터 가져오기
    floors = floors_data.get(selected_building, {})

    # # 데이터 필터링 (동에 맞는 데이터만 로드)
    # filtered_status = [
    #     row for row in status if row[0] == int(selected_building)
    # ]

    # 데이터 추가 (1층부터 25층까지)
    for floor_group in range(1, 26, 5):  # 5층 단위로 그룹화
        group_key = f"{floor_group}-{floor_group + 4}"
        floors_data["101"][group_key] = []
        if group_key not in floors_ho:
            floors_ho[group_key] = []

        for i in range(floor_group, floor_group + 5):  # 각 그룹 내 1층씩 처리
            floor_rooms = []
            floor_hos = []  # 각 층의 호수 리스트
            for j in range(1, 3):  # 방 2개씩
                ho = i * 100 + j  # 호수 계산 (101, 102, ..., 2502)
                room_status = next(
                    (row[2] for row in status if int(row[0]) == 101 and int(row[1]) == ho),
                    "normal"
                )
                floor_rooms.append(room_status)
                floor_hos.append(str(ho))
            
            # 층 데이터 추가
            floors_data["101"][group_key].append(floor_rooms)
            floors_ho[group_key].append(floor_hos)

    # combined_floors 생성 (필터링된 데이터 활용)
    # floors와 floors_ho를 개별적으로 분리
    combined_floors = {
        floor_number: [
            {"status": status, "ho": ho}
            for status_list, ho_list in zip(floor_status, floor_ho)
            for status, ho in zip(status_list, ho_list)
        ]
        for floor_number, (floor_status, floor_ho) in zip(floors.keys(), zip(floors.values(), floors_ho.values()))
    }

    '''
    오른쪽 박스 상태별 데이터
    '''
    # summary 만들기
    summary = {
        "danger": 0,
        "caution": 0,
        "normal": 0
    }
    for condition, count in rightbox:
        summary[condition] = count

    # 위험 상태 정보
    danger_info = {
        "dong": danger_result[0][0],
        "ho": danger_result[0][1]
    } if danger_result else None

    return render_template(
        "admin_mainpage.html",
        predict_elec=predict_electric,
        predict_water=predict_water_original, 
        output1=output1, 
        output2=output2, 
        ElecErrorMargin=ElecErrorMargin,
        WaterErrorMargin=WaterErrorMargin,
        building_status=building_status,
        combined_floors=combined_floors,
        selected_building=selected_building, # 선택된 동번호 전달
        floors=floors, # 층별 정보 전달
        resident_data=resident_data, # 거주자 데이터 전달
        floors_ho = floors_ho, # 호수 정보 전달
        status = status, # from DB
        summary = summary, # 전체 동 summary
        danger_info = danger_info, # 위험 상태 가져오기
        announcement = announcement_data,
        board_data=board_data
    )

# http://127.0.0.1:5000/customer_mainpage
@APP.route("/customer_mainpage")
def customer_mainpage():
    
    try:
        # 데이터베이스 연결
        cur1 = conn.cursor()
        query1 = "SELECT CAST(water_usage AS FLOAT) FROM water WHERE member_id = 'user010'"
        cur1.execute(query1)
        rows1 = cur1.fetchall()
        cur1.close()

        cur2 = conn.cursor()
        query2 = "SELECT CAST(electric_usage AS FLOAT) FROM electric WHERE member_id = 'user010'"
        cur2.execute(query2)
        rows2 = cur2.fetchall()
        cur2.close()

        # 최신 공지사항 1개 가져오기 (내림차순 정렬)
        cur3 = conn.cursor()
        query3 = "SELECT announcement_id, announcement_title, announcement_content, announcement_date FROM announcement ORDER BY announcement_id DESC LIMIT 1"
        cur3.execute(query3)
        announcement = cur3.fetchall()
        cur3.close()


        # 게시판에 최신 5개 데이터 가져오기
        cur4 = conn.cursor()
        query4 = "SELECT board_id, board_title, board_author, board_date FROM board ORDER BY board_id DESC LIMIT 5"
        cur4.execute(query4)
        rows4 = cur4.fetchall()
        cur4.close()


        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        water_model_path = os.path.join(BASE_DIR, "models/water_autoencoder_model.pth")
        electric_model_path = os.path.join(BASE_DIR, "models/electric_autoencoder_model.pth")

        water_scaler_path = os.path.join(BASE_DIR, "models/water_mm_scaler.pkl")

        # 데이터 추출
        output1 = [row[0] for row in rows1[-28:]]  # water_usage 변환
        output2 = [row[0] for row in rows2[-28:]]  # electric_usage 변환

        output1 = output1[-28:]
        output2 = output2[-28:]

        ElecErrorMargin = []

        for elec in output2:
            if elec <= 1.4:
                ElecErrorMargin.append(0.08)
            elif elec <= 1.6:
                ElecErrorMargin.append(0.13)
            elif elec <= 2:
                ElecErrorMargin.append(0.15)
        
        WaterErrorMargin = []

        for water in output1:
            WaterErrorMargin.append(40)

        water_ts, water_scaler = preprocessing(output1, water_scaler_path)
        elec_ts = torch.FloatTensor([output2])

        water_model = load_water_model(water_model_path)
        electric_model = load_electric_model(electric_model_path)

        predict_electric = electric_model(elec_ts).squeeze(0).tolist()
        predict_water = water_model(water_ts)

        predict_water_original = water_scaler.inverse_transform(predict_water.detach().numpy()).squeeze(0).tolist()

        # 공지사항이 없으면 None을 전달하고, 있으면 공지사항 데이터를 전달
        if announcement:
            announcement_data = {
                'idx' : announcement[0][0],
                'title': announcement[0][1],
                'content': announcement[0][2],
                'date': announcement[0][3]
            }
        else:
            announcement_data = None
        
        # if board_data:
        #     board_data = [{
        #         'id': board[0],
        #         'title': board[1],
        #         'author': board[2],
        #         'date': board[3]
        #     } for board in board_data]
        # else:
        #     board_data = None

        # 달력 부분
        today = datetime.today()

        # 기본값: 현재 연도와 6월로 설정
        year = request.args.get('year', today.year, type=int)
        # month = request.args.get('month', today.month, type=int)
        month = request.args.get('month', 6, type=int)

        # 달력 생성
        cal = calendar.Calendar(firstweekday=6)  # 6: 일요일 시작
        month_days = cal.monthdayscalendar(year, month)

        '''
        달력 마지막 주 다음달 추가
        '''
        
        # # 다음 달의 연도 및 월 계산
        # if month == 12:
        #     next_month = 1
        #     next_year = year + 1
        # else:
        #     next_month = month + 1
        #     next_year = year

        # 현재 주 (최근 일주일: 월요일~일요일)
        current_week = [
            (today - timedelta(days=today.weekday() - i)).day for i in range(7)
        ]

        # # 다음 달의 첫 주 가져오기
        # next_month_days = cal.monthdayscalendar(next_year, next_month)

        # # 현재 달의 마지막 주가 다음 달의 첫 주와 겹치는 경우 처리
        # last_week = month_days[-1]
        # if 0 in last_week:  # 마지막 주에 빈 날짜(0)가 있다면 다음 달로 넘어가는 주
        #     first_week_next_month = next_month_days[0]
        #     for i, day in enumerate(last_week):
        #         if day == 0:  # 현재 달의 빈칸을 다음 달의 날짜로 채움
        #             last_week[i] = first_week_next_month.pop(0)

        # # 병합 후 달력 데이터 업데이트
        # month_days[-1] = last_week

        cnt = 0
        for i in range(4):
            if not (predict_water_original[23 + i] - WaterErrorMargin[23 + i]) <= output1[23 + i] <= (predict_water_original[23 + i] + WaterErrorMargin[23 + i]):
                if not (predict_electric[23 + i] - ElecErrorMargin[23 + i]) <= output2[23 + i] <= (predict_electric[23 + i] + ElecErrorMargin[23 + i]):
                    cnt += 1

        # 템플릿에 공지사항 데이터 추가하여 렌더링
        return render_template("/customer_mainpage.html", 
                       predict_elec=predict_electric,
                       predict_water=predict_water_original, 
                       output1=output1, 
                       output2=output2, 
                       ElecErrorMargin=ElecErrorMargin,
                       WaterErrorMargin=WaterErrorMargin,
                       announcement=announcement_data, 
                       board_data=rows4,
                       year=year, # 연 (달력)
                       month=month, # 월 (달력)
                    #    next_year=next_year,
                    #    next_month=next_month,
                       month_days=month_days,
                       current_week=current_week,
                       cnt = cnt
                       )
    except Exception as e:
        print(f"Error: {e}")
        return render_template('customer_mainpage.html', message="오류가 발생했습니다.")

# http://127.0.0.1:5000/register
@APP.route("/register")
def register():
    return render_template("/register.html")

# http://127.0.0.1:5000/welcome_admin
@APP.route("/welcome_admin", methods=['GET', 'POST'])
def welcome_admin():
    return render_template("/welcome_admin.html")

# http://127.0.0.1:5000/welcome_customer
@APP.route("/welcome_customer", methods=['GET', 'POST'])
def welcome_customer():
    return render_template("/welcome_customer.html")

# http://127.0.0.1:5000/customer_membership
@APP.route("/customer_membership", methods=['GET', 'POST'])
def customer_membership():
    if request.method == 'POST':
        # 폼 데이터 수집
        
        userid = request.form.get('userid') # 보호자 id
        password = request.form.get('password') # 보호자 pw
        guardian_name = request.form.get('guardian-name') # 보호자 이름
        address = request.form.get('address') # 보호자 주소
        
        phone_prefix = request.form.get('phone_prefix')
        phone2 = request.form.get('phone2')
        phone3 = request.form.get('phone3')

        guardian_resident1 = request.form.get('guardian-resident1')
        guardian_resident2 = request.form.get('guardian-resident2')
        guardian_resident_number = f"{guardian_resident1}-{guardian_resident2}"

        guardian_phone_number = f"{phone_prefix}-{phone2}-{phone3}"

        # 주민등록번호 및 전화번호 조합 (피보호자)
        name = request.form.get('name')
        resident1 = request.form.get('resident1')
        resident2 = request.form.get('resident2')

        phone_prefix2 = request.form.get('phone_prefix2')
        phone4 = request.form.get('phone4')
        phone5 = request.form.get('phone5')

        resident_number = f"{resident1}-{resident2}"
        phone_number = f"{phone_prefix2}-{phone4}-{phone5}"

        # DB 저장
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = """
                INSERT INTO member (
                    member_id,
                    member_password,
                    registration_number,
                    member_address,
                    member_phone_number,
                    managed_entity_name,
                    managed_entity_registration_number,
                    managed_entity_phone_number,
                    member_name
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                userid, 
                password, 
                resident_number, 
                address, 
                phone_number, 
                name, 
                guardian_resident_number,
                guardian_phone_number,
                guardian_name
            ))
            conn.commit()
            cursor.close()
            conn.close()
            return redirect(url_for('welcome_customer'))
        
        except pymysql.MySQLError as e:
            print(f"Database error: {e}")

    return render_template("/customer_membership.html")

# http://127.0.0.1:5000/admin_membership
@APP.route("/admin_membership", methods=['GET', 'POST'])
def admin_membership():
    if request.method == 'POST':
        # 폼 데이터 수집
        username = request.form.get('username')
        password = request.form.get('password')
        apt_name = request.form.get('apt-name')
        apt_address = request.form.get('apt-address')
        apt_code = request.form.get('apt-codes')

        # DB 저장
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # office 테이블에 삽입
            query_office = """
                INSERT INTO office (
                office_id,
                password,
                apartment_name,
                apartment_address,
                managed_code
            )
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query_office, 
                           (username, password, apt_name, apt_address, apt_code))

            conn.commit()
            cursor.close()
            conn.close()
            return redirect(url_for('welcome_admin'))
        except pymysql.MySQLError as e:
            print(f"Database error: {e}")
        
    return render_template("/admin_membership.html")

# http://127.0.0.1:5000/board_admin
@APP.route("/board_admin")
def board_admin():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        query = "SELECT board_id, board_title, board_author, board_date FROM board ORDER BY board_date DESC"
        cursor.execute(query)
        boards = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template("board_admin.html", boards=boards)

    except pymysql.MySQLError as e:
        print(f"Database error: {e}")
        return "데이터베이스 오류가 발생했습니다.", 500

# http://127.0.0.1:5000/board_customer
@APP.route("/board_customer")
def board_customer():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # 게시글 리스트 가져오기
        query = "SELECT board_id, board_title, board_author, board_date FROM board ORDER BY board_date DESC"
        cursor.execute(query)
        boards = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template("board_customer.html", boards=boards)

    except pymysql.MySQLError as e:
        print(f"Database error: {e}")
        return "데이터베이스 오류가 발생했습니다.", 500

@APP.route('/board_content_admin/<int:board_id>')
def board_content_admin(board_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # 해당 게시물의 상세 내용 가져오기
        query = """
            SELECT board_title, board_author, board_date, board_content
            FROM board
            WHERE board_id = %s
        """
        cursor.execute(query, (board_id,))
        board = cursor.fetchone()

        cursor.close()
        conn.close()

        if board:
            # 날짜 포맷을 처리하여 템플릿에 넘기기
            board['board_date'] = board['board_date'].strftime('%Y-%m-%d')  # 날짜 형식을 '%Y-%m-%d'로 변경
            return render_template('board_content_admin.html', board=board)
        else:
            return "게시글을 찾을 수 없습니다.", 404

    except pymysql.MySQLError as e:
        print(f"Database error: {e}")
        return "데이터베이스 오류가 발생했습니다.", 500
    
@APP.route('/board_content_customer/<int:board_id>')
def board_content_customer(board_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # 해당 게시물의 상세 내용 가져오기
        query = """
            SELECT board_title, board_author, board_date, board_content
            FROM board
            WHERE board_id = %s
        """
        cursor.execute(query, (board_id,))
        board = cursor.fetchone()

        board['board_id'] = board_id

        cursor.close()
        conn.close()

        if board:
            # 날짜 포맷을 처리하여 템플릿에 넘기기
            board['board_date'] = board['board_date'].strftime('%Y-%m-%d')  # 날짜 형식을 '%Y-%m-%d'로 변경
            return render_template('board_content_customer.html', board=board)
        else:
            return "게시글을 찾을 수 없습니다.", 404

    except pymysql.MySQLError as e:
        print(f"Database error: {e}")
        return "데이터베이스 오류가 발생했습니다.", 500

# http://127.0.0.1:5000/notice_admin
@APP.route("/notice_admin")
def notice_admin():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        query = "SELECT announcement_id, announcement_title, announcement_author, announcement_date FROM announcement ORDER BY announcement_date DESC"
        cursor.execute(query)
        announcement = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template("notice_admin.html", announcement=announcement)

    except pymysql.MySQLError as e:
        print(f"Database error: {e}")
        return "데이터베이스 오류가 발생했습니다.", 500

# http://127.0.0.1:5000/notice_customer
@APP.route('/notice_customer', methods=['GET'])
def notice_customer():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # 공지사항 리스트 가져오기
        cursor.execute("SELECT * FROM announcement ORDER BY announcement_date DESC")
        announcement = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template("notice_customer.html", announcement=announcement)

    except pymysql.MySQLError as e:
        print(f"Database error: {e}")
        return "데이터베이스 오류가 발생했습니다.", 500

# http://127.0.0.1:5000/write_board_admin
@APP.route('/write_board_admin', methods=['GET', 'POST'])
def write_board_admin():
    if request.method == 'POST':
        title = request.form.get('board_title')  # 제목
        content = request.form.get('board_content')  # 내용

        # 필수 입력값 확인
        if not title or not content:
            return "제목과 내용을 입력하세요.", 400

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 데이터 삽입 쿼리 (board_id는 자동 증가, board_author는 '관리자', board_date는 현재 날짜)
            query = """
                INSERT INTO board (board_title, board_author, board_date, board_content)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (title, "관리자", datetime.now().date(), content))  # 관리자로 작성자 설정
            conn.commit()

            cursor.close()
            conn.close()

            return redirect(url_for('board_admin'))

        except Exception as e:
            print(f"Database error: {e}")
            return "데이터베이스 오류가 발생했습니다.", 500

    return render_template('write_board_admin.html')

# http://127.0.0.1:5000/write_board_customer
@APP.route('/write_board_customer', methods=['GET', 'POST'])
def write_board_customer():
    if request.method == 'POST':
        # 폼 데이터 처리
        title = request.form.get('board_title')  # 제목
        content = request.form.get('board_content')  # 내용

        # 필수 입력값 확인
        if not title or not content:
            return "제목과 내용을 입력하세요.", 400

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 데이터 삽입 쿼리 (board_id는 자동 증가, board_author는 '고객', board_date는 현재 날짜)
            query = """
                INSERT INTO board (board_title, board_author, board_date, board_content)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (title, "개인", datetime.now().date(), content))  # 고객으로 작성자 설정
            conn.commit()

            cursor.close()
            conn.close()

            # 게시글 작성 후 게시판 목록 페이지로 리다이렉트
            return redirect(url_for('board_customer'))

        except Exception as e:
            print(f"Database error: {e}")
            return "데이터베이스 오류가 발생했습니다.", 500

    # GET 요청의 경우 게시판 글쓰기 폼 렌더링
    return render_template('write_board_customer.html')

# http://127.0.0.1:5000/write_notice
@APP.route('/write_notice', methods=['GET', 'POST'])
def write_notice():
    if request.method == "POST":
        title = request.form.get("announcement_title")
        content = request.form.get("announcement_content")
        date = datetime.now().date()  # 현재 시간 자동 입력
        author = "관리자"  # 글쓴이를 "관리자"로 설정

        if not title or not content:
            return "제목과 내용을 모두 입력해야 합니다.", 400

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 공지사항 저장 쿼리 실행 (announcement_id는 자동으로 처리되므로 제외)
            query = """
                INSERT INTO announcement (announcement_title, announcement_content, announcement_date, announcement_author)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (title, content, date, author))  # "관리자" 값을 포함
            conn.commit()

            cursor.close()
            conn.close()

            # 저장 후 공지사항 리스트 페이지로 리다이렉트
            return redirect(url_for("notice_admin"))

        except pymysql.MySQLError as e:
            print(f"Database error: {e}")
            return "데이터베이스 오류가 발생했습니다.", 500

    # GET 요청 시 공지사항 작성 페이지 렌더링
    return render_template("write_notice.html")

@APP.route('/notice_content_admin/<int:announcement_id>')
def notice_content_admin(announcement_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # 해당 공지사항의 상세 내용 가져오기
        query = """
            SELECT announcement_title, announcement_author, announcement_date, announcement_content
            FROM announcement
            WHERE announcement_id = %s
        """
        cursor.execute(query, (announcement_id,))
        announcement = cursor.fetchone()

        announcement['announcement_id'] = announcement_id  # announcement_id 추가

        cursor.close()
        conn.close()

        if announcement:
            # 날짜 포맷을 처리하여 템플릿에 넘기기
            announcement['announcement_date'] = announcement['announcement_date'].strftime('%Y-%m-%d')  # 날짜 형식을 '%Y-%m-%d'로 변경
            return render_template('notice_content_admin.html', announcement=announcement)
        else:
            return "공지사항을 찾을 수 없습니다.", 404

    except pymysql.MySQLError as e:
        print(f"Database error: {e}")
        return "데이터베이스 오류가 발생했습니다.", 500



@APP.route('/notice_content_customer/<int:announcement_id>')
def notice_content_customer(announcement_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT announcement_content, announcement_date, announcement_title, announcement_author FROM announcement WHERE announcement_id = %s", (announcement_id,))
    announcement_content = cursor.fetchone()
    cursor.close()
    connection.close()
    return render_template('notice_content_customer.html', content=announcement_content)

# 고객 게시판 삭제 라우트
@APP.route('/delete_board_customer/<int:board_id>', methods=['GET'])
def delete_board_customer(board_id):
    try:
        # 데이터베이스 연결
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 게시글 삭제 쿼리
        delete_query = "DELETE FROM board_customer WHERE board_id = %s"
        cursor.execute(delete_query, (board_id,))
        
        # 변경사항 커밋
        conn.commit()
        
        # 커서와 연결 닫기
        cursor.close()
        conn.close()
        
        # 삭제 후 게시판 페이지로 리다이렉트
        flash('고객 게시판 게시글이 성공적으로 삭제되었습니다.', 'success')
        return redirect(url_for('board_customer'))
    
    except Exception as e:
        # 오류 처리
        flash(f'고객 게시판 게시글 삭제 중 오류가 발생했습니다: {str(e)}', 'error')
        return redirect(url_for('board_customer'))


# 조건부 실행 ---------------------------------------------------------
if __name__ == '__main__':
    
    APP.run()