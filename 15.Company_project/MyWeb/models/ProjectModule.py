import os
import torch
import torch.nn as nn
import pickle
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics.regression import R2Score, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError


class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, input_size, n_layers, dropout,
                 bidirectional):
        super().__init__()

        self.model = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_dim,
            num_layers = n_layers,
            dropout = dropout,
            bidirectional = bidirectional,
            batch_first = True
        )

        if bidirectional:
            self.linear = nn.Linear(hidden_dim * 2, 1)
        
        else:
            self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        output, _ = self.model(inputs)
        logits = self.linear(output)

        return logits

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

def predict_values(water_model, electric_model, water_TS, electric_TS):

    water_predict_value = water_model(water_TS)
    electric_predict_value = electric_model(electric_TS)

    return water_predict_value.item(), electric_predict_value.item()


def split_tuple(tuple, member):
    L = []
    L2 = []

    for i in range(len(tuple)):
        if i % (len(tuple) // member) == 0 and i != 0:
            L.append(L2)
            L2 = []

        L2.append(tuple[i][1])

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
