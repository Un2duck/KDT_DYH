import pymysql
import torch
import torch.nn as nn
import os
import pandas as pd
import pickle

def get_db_connection():
    return pymysql.connect(
        host='172.20.146.27',
        user='younghun',
        password='1234',
        db='quasar_copy',
        charset='utf8'
    )

conn = get_db_connection()

def split_tuple(tuple, member):
    L = []
    L2 = []

    for i in range(len(tuple)):
        if i % (len(tuple) // member) == 0 and i != 0:
            L.append(L2)
            L2 = []

        # tuple[i][1] 값을 L2에 추가
        L2.append(tuple[i][1])

    # 마지막 남은 L2 리스트 추가
    L.append(L2)

    return L

def split_tuple2(tuple, member):
    L = []
    L2 = []

    for i in range(len(tuple)):
        if i % (len(tuple) // member) == 0 and i != 0:
            L.append(L2)
            L2 = []

        # tuple[i][1] 값을 L2에 추가
        L2.append(tuple[i][0])

    # 마지막 남은 L2 리스트 추가
    L.append(L2)

    return L


def preprocessing(inputs, scaler_path):
    dataframe = pd.DataFrame(columns = range(28))
    dataframe.loc[0] = inputs

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    scaled_dataframe = scaler.transform(dataframe)

    featureTS = torch.FloatTensor(scaled_dataframe)

    return featureTS, scaler

def load_water_model(model_file_path):
    input_size = 28
    latent_dim = 16
    n_layers = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    water_lstm_model = LSTMAutoEncoderModel(input_size = input_size, latent_dim = latent_dim, 
                             n_layers = n_layers).to(DEVICE)
    
    water_lstm_model.load_state_dict(torch.load(model_file_path, map_location = torch.device(DEVICE), weights_only = True))
    water_lstm_model.eval()

    return water_lstm_model

def load_electric_model(model_file_path):
    input_size = 28
    latent_dim = 16
    n_layers = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    electric_lstm_model = LSTMAutoEncoderModel(input_size = input_size, latent_dim = latent_dim, 
                             n_layers = n_layers).to(DEVICE)
    
    electric_lstm_model.load_state_dict(torch.load(model_file_path, map_location = torch.device(DEVICE), weights_only = True))
    electric_lstm_model.eval()

    return electric_lstm_model

class LSTMAutoEncoderModel(nn.Module):
    def __init__(self, input_size, latent_dim, n_layers):
        super().__init__()

        self.encoder = nn.GRU(
            input_size = input_size,
            hidden_size = latent_dim,
            num_layers = n_layers,
            batch_first = True
        )

        # self.latent_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.GRU(
            input_size = latent_dim,
            hidden_size = input_size,
            num_layers = n_layers,
            batch_first = True
        )

        self.output_layer = nn.Linear(input_size, input_size)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)

        _, encoder = self.encoder(inputs)

        if self.encoder.bidirectional:
            encoder = torch.cat([encoder[-2], encoder[-1]], dim = -1)
        else:
            encoder = encoder[-1]

        decoder, _ = self.decoder(encoder)

        reconstruction = self.output_layer(decoder.squeeze(1))

        return reconstruction
    

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
        l = sorted(s)

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
    


print(building_status)