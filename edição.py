import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
import os
import joblib
import warnings
from datetime import datetime
from scipy import stats
import traceback

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class StatisticalAnalyzer:
    def __init__(self, historical_data):
        self.data = historical_data
        self.num_numbers = 25
        self.numbers_to_choose = 15
        
    def get_frequency_metrics(self, window_size=30):
        """Analisa frequências em diferentes janelas temporais"""
        frequencies = {}
        recency = {}
        for window in [window_size, window_size*2, window_size*4]:
            last_n = self.data.tail(window)
            freq = {}
            for i in range(1, self.num_numbers + 1):
                freq[i] = sum((last_n[[f'Bola{j}' for j in range(1, 16)]].values == i).any(axis=1))
            frequencies[window] = freq
            
        for num in range(1, self.num_numbers + 1):
            for idx in range(len(self.data)-1, -1, -1):
                if num in [self.data.iloc[idx][f'Bola{i}'] for i in range(1, 16)]:
                    recency[num] = len(self.data) - idx - 1
                    break
        
        return frequencies, recency

    def get_pattern_metrics(self):
        """Analisa padrões de sequências e combinações"""
        patterns = {
            'consecutive_pairs': {},
            'common_combinations': {},
            'position_frequencies': {}
        }
        
        for idx in range(len(self.data)):
            numbers = sorted([self.data.iloc[idx][f'Bola{i}'] for i in range(1, 16)])
            for i in range(len(numbers)-1):
                pair = (numbers[i], numbers[i+1])
                patterns['consecutive_pairs'][pair] = patterns['consecutive_pairs'].get(pair, 0) + 1
        
        for pos in range(1, 16):
            patterns['position_frequencies'][pos] = {}
            for num in range(1, self.num_numbers + 1):
                patterns['position_frequencies'][pos][num] = sum(self.data[f'Bola{pos}'] == num)
        
        return patterns

    def get_correlation_metrics(self):
        """Analisa correlações entre números"""
        correlations = np.zeros((self.num_numbers, self.num_numbers))
        number_matrix = np.zeros((len(self.data), self.num_numbers))
        
        for idx, row in self.data.iterrows():
            numbers = [int(row[f'Bola{i}']) - 1 for i in range(1, 16)]
            number_matrix[idx, numbers] = 1
            
        correlations = np.corrcoef(number_matrix.T)
        return correlations

    def get_trend_metrics(self, window_size=10):
        """Analisa tendências recentes"""
        trends = {}
        recent_data = self.data.tail(window_size)
        
        for num in range(1, self.num_numbers + 1):
            appearances = []
            for idx in range(len(recent_data)):
                appeared = 1 if num in [recent_data.iloc[idx][f'Bola{i}'] for i in range(1, 16)] else 0
                appearances.append(appeared)
            
            trend = np.polyfit(range(len(appearances)), appearances, 1)[0]
            trends[num] = trend
            
        return trends

    def calculate_number_weights(self):
        """Calcula pesos para cada número baseado em todas as métricas"""
        weights = np.zeros(self.num_numbers)
        
        frequencies, recency = self.get_frequency_metrics()
        patterns = self.get_pattern_metrics()
        correlations = self.get_correlation_metrics()
        trends = self.get_trend_metrics()
        
        for i in range(self.num_numbers):
            freq_weight = sum(frequencies[k][i+1] for k in frequencies.keys()) / len(frequencies)
            recency_weight = 1 / (recency[i+1] + 1)
            trend_weight = trends[i+1]
            correlation_weight = np.mean(correlations[i])
            
            weights[i] = (
                0.3 * freq_weight +
                0.25 * recency_weight +
                0.25 * trend_weight +
                0.2 * correlation_weight
            )
        
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        return weights

class LotteryPredictor:
    def __init__(self):
        self.data_file = 'dados.txt'
        self.lstm = None
        self.rf = None
        self.xgb_models = None
        self.models_dir = '.models/'
        self.data_dir = '.data/'
        self.sequence_length = 15
        self.num_numbers = 25
        self.numbers_to_choose = 15
        
        for directory in [self.models_dir, self.data_dir]:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    if os.name == 'nt':
                        os.system(f'attrib +h "{directory}"')
                except Exception as e:
                    print(f"Erro ao criar diretório {directory}: {e}")
        
        self.historical_data = self._load_data('historical.pkl')
        self.predictions = self._load_data('predictions.pkl')
        self.load_models()
        self.analyzer = None if self.historical_data.empty else StatisticalAnalyzer(self.historical_data)

    def _load_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            return joblib.load(filepath)
        return pd.DataFrame()

    def _save_data(self, data, filename):
        try:
            filepath = os.path.join(self.data_dir, filename)
            joblib.dump(data, filepath)
        except Exception as e:
            print(f"Erro ao salvar dados: {e}")

    def load_models(self):
        try:
            if all(os.path.exists(os.path.join(self.models_dir, f)) for f in ['lstm.h5', 'rf.pkl', 'xgb_models.pkl']):
                self.lstm = load_model(os.path.join(self.models_dir, 'lstm.h5'))
                self.rf = joblib.load(os.path.join(self.models_dir, 'rf.pkl'))
                self.xgb_models = joblib.load(os.path.join(self.models_dir, 'xgb_models.pkl'))
                return True
            return False
        except Exception as e:
            print(f"Erro ao carregar modelos: {e}")
            return False

    def save_models(self):
        try:
            self.lstm.save(os.path.join(self.models_dir, 'lstm.h5'))
            joblib.dump(self.rf, os.path.join(self.models_dir, 'rf.pkl'))
            joblib.dump(self.xgb_models, os.path.join(self.models_dir, 'xgb_models.pkl'))
        except Exception as e:
            print(f"Erro ao salvar modelos: {e}")

    def process_new_data(self):
        if not os.path.exists(self.data_file) or os.path.getsize(self.data_file) == 0:
            return False

        try:
            new_data = pd.read_csv(self.data_file, sep='\t')
            if not new_data.empty:
                new_data['Data Sorteio'] = pd.to_datetime(new_data['Data Sorteio'], format='%d/%m/%Y')
                
                if not self.historical_data.empty:
                    new_data = new_data[new_data['Concurso'] > self.historical_data['Concurso'].max()]
                
                if not new_data.empty:
                    self.historical_data = pd.concat([self.historical_data, new_data])
                    self._save_data(self.historical_data, 'historical.pkl')
                    self.analyzer = StatisticalAnalyzer(self.historical_data)
                    open(self.data_file, 'w').close()
                    return True
            return False
        except Exception as e:
            print(f"Erro ao processar novos dados: {e}")
            return False

    def prepare_sequences(self):
        if len(self.historical_data) < self.sequence_length:
            return None, None

        data = np.zeros((len(self.historical_data), self.num_numbers))
        for idx, row in self.historical_data.iterrows():
            numbers = [int(row[f'Bola{i}']) - 1 for i in range(1, 16)]
            data[idx, numbers] = 1

        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])

        return np.array(X), np.array(y)

    def train_models(self):
        X, y = self.prepare_sequences()
        if X is None or y is None:
            return False

        try:
            print("Treinando LSTM...")
            self.lstm = Sequential([
                Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.sequence_length, self.num_numbers)),
                Dropout(0.4),
                Bidirectional(LSTM(64)),
                Dropout(0.4),
                Dense(256, activation='relu'),
                Dropout(0.4),
                Dense(self.num_numbers, activation='sigmoid')
            ])
            self.lstm.compile(optimizer='adam', loss='binary_crossentropy')
            
            early_stopping = EarlyStopping(monitor='loss', patience=5)
            self.lstm.fit(X, y, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
            
            print("Treinando Random Forest...")
            X_flat = X.reshape(X.shape[0], -1)
            self.rf = []
            for i in range(self.num_numbers):
                rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, n_jobs=-1)
                rf_model.fit(X_flat, y[:, i])
                self.rf.append(rf_model)
            
            print("Treinando XGBoost...")
            self.xgb_models = []
            for i in range(self.num_numbers):
                model = xgb.XGBClassifier(objective='binary:logistic', max_depth=6, learning_rate=0.1, n_estimators=200)
                model.fit(X_flat, y[:, i])
                self.xgb_models.append(model)
            
            print("Salvando modelos...")
            self.save_models()
            return True
        except Exception as e:
            print(f"Erro no treinamento: {e}")
            traceback.print_exc()
            return False

    def generate_prediction(self):
        if self.historical_data.empty:
            print("Não há dados históricos disponíveis.")
            return None

        try:
            X, _ = self.prepare_sequences()
            if X is None:
                return None

            last_sequence = X[-1:]
            last_sequence_flat = last_sequence.reshape(1, -1)

            # Previsão LSTM
            lstm_pred = self.lstm.predict(last_sequence, verbose=0)[0]
            lstm_pred = np.array(lstm_pred).flatten()[:self.num_numbers]

            # Previsão Random Forest
            rf_pred = np.zeros(self.num_numbers)
            for i in range(min(len(self.rf), self.num_numbers)):
                rf_model = self.rf[i]
                try:
                    proba = rf_model.predict_proba(last_sequence_flat)
                    rf_pred[i] = proba[0, 1] if proba.shape[1] >= 2 else proba[0, 0]
                except:
                    try:
                        pred = rf_model.predict(last_sequence_flat)
                        rf_pred[i] = pred[0] if isinstance(pred, np.ndarray) else float(pred)
                    except:
                        rf_pred[i] = 0.0

            # Previsão XGBoost
            xgb_pred = np.zeros(self.num_numbers)
            for i in range(min(len(self.xgb_models), self.num_numbers)):
                model = self.xgb_models[i]
                try:
                    proba = model.predict_proba(last_sequence_flat)
                    xgb_pred[i] = proba[0, 1] if proba.shape[1] >= 2 else proba[0, 0]
                except:
                    try:
                        pred = model.predict(last_sequence_flat)
                        xgb_pred[i] = pred[0] if isinstance(pred, np.ndarray) else float(pred)
                    except:
                        xgb_pred[i] = 0.0

            # Incorporar análise estatística
            if self.analyzer:
                statistical_weights = self.analyzer.calculate_number_weights()
            else:
                statistical_weights = np.ones(self.num_numbers) / self.num_numbers

            # Normalização com ruído aleatório
            def safe_normalize(arr):
                arr = np.array(arr).flatten()[:self.num_numbers]
                arr += np.random.normal(0, 0.01, arr.shape)
                min_val = np.min(arr)
                max_val = np.max(arr)
                if max_val - min_val < 1e-10:
                    return np.random.random(arr.shape)
                return (arr - min_val) / (max_val - min_val)

            lstm_pred = safe_normalize(lstm_pred)
            rf_pred = safe_normalize(rf_pred)
            xgb_pred = safe_normalize(xgb_pred)
            statistical_weights = safe_normalize(statistical_weights)

            # Combinação das previsões com pesos
            weights = np.random.dirichlet(np.ones(4))
            combined = (weights[0] * lstm_pred + 
                       weights[1] * rf_pred + 
                       weights[2] * xgb_pred +
                       weights[3] * statistical_weights)

            numbers = np.argsort(combined)[-self.numbers_to_choose:] + 1
            numbers = sorted(numbers)
            # Adiciona aleatoriedade se os números forem iguais ao último sorteio
            last_numbers = sorted([int(self.historical_data.iloc[-1][f'Bola{i}']) for i in range(1, 16)])
            if numbers == last_numbers:
                num_to_replace = np.random.randint(1, 4)
                indices_to_replace = np.random.choice(len(numbers), num_to_replace, replace=False)
                new_numbers = list(set(range(1, 26)) - set(numbers))
                for idx in indices_to_replace:
                    numbers[idx] = np.random.choice(new_numbers)
                numbers = sorted(numbers)

            # Salvando a previsão
            prediction = {
                'data': datetime.now(),
                'numeros': numbers,
                'concurso': self.historical_data['Concurso'].max() + 1
            }
            
            self.predictions = pd.concat([self.predictions, pd.DataFrame([prediction])])
            self._save_data(self.predictions, 'predictions.pkl')

            return numbers
        except Exception as e:
            print(f"Erro na geração de previsão: {e}")
            traceback.print_exc()
            return None

    def run(self):
        print("\n=== Sistema de Previsão da Lotofácil ===")
        
        has_new_data = self.process_new_data()
        models_exist = self.load_models()

        if has_new_data or not models_exist:
            print("\nIniciando treinamento dos modelos...")
            if not self.train_models():
                print("Erro no treinamento dos modelos.")
                return

        print("\nGerando previsão...")
        numbers = self.generate_prediction()
        
        if numbers is not None:
            print("\nNúmeros sugeridos para o próximo sorteio:")
            print(numbers)
            
            if not self.historical_data.empty and self.analyzer:
                # Análise estatística adicional
                frequencies, recency = self.analyzer.get_frequency_metrics()
                trends = self.analyzer.get_trend_metrics()
                
                print("\nAnálise dos últimos 30 sorteios:")
                print("Números mais frequentes:")
                last_30_freq = frequencies[30]
                sorted_freq = sorted(last_30_freq.items(), key=lambda x: x[1], reverse=True)
                for num, freq in sorted_freq[:5]:
                    trend_direction = "↑" if trends[num] > 0 else "↓" if trends[num] < 0 else "→"
                    print(f"Número {num}: {freq} vezes {trend_direction}")
                
                print("\nNúmeros menos sorteados recentemente:")
                sorted_recency = sorted(recency.items(), key=lambda x: x[1], reverse=True)
                for num, rec in sorted_recency[:3]:
                    print(f"Número {num}: {rec} sorteios sem aparecer")
        else:
            print("Não foi possível gerar uma previsão.")

def main():
    predictor = LotteryPredictor()
    predictor.run()

if __name__ == "__main__":
    main()