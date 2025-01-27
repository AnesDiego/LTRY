import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import entropy, chi2_contingency
from scipy.special import softmax
import xgboost as xgb
import tensorflow as tf
import os
import joblib
import warnings
from datetime import datetime, timedelta
import itertools
from collections import defaultdict, Counter
import logging
import json
import traceback

# PRIMEIRO: Criar diretório .data ANTES de qualquer configuração de logging
data_dir = '.data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# DEPOIS: Configurar warnings e logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# FINALMENTE: Configurar logging
log_file = os.path.join(data_dir, 'lottery_predictor.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Verificação adicional (opcional, apenas para debug)
logger.info(f"Diretório .data existe: {os.path.exists(data_dir)}")
logger.info(f"Arquivo de log criado em: {log_file}")

class MarkovChain:
    """
    Implementa uma Cadeia de Markov para análise de transições de números
    """
    def __init__(self, n_numbers=25):
        self.n_numbers = n_numbers
        self.transition_matrix = np.zeros((n_numbers, n_numbers))
        self.state_counts = np.zeros(n_numbers)
        
    def fit(self, sequences):
        """
        Treina a cadeia de Markov com sequências de números
        """
        for sequence in sequences:
            for i in range(len(sequence)-1):
                current = sequence[i] - 1  # Ajusta para índice 0-based
                next_num = sequence[i+1] - 1
                self.transition_matrix[current][next_num] += 1
                self.state_counts[current] += 1
        
        # Normaliza as probabilidades
        for i in range(self.n_numbers):
            if self.state_counts[i] > 0:
                self.transition_matrix[i] = self.transition_matrix[i] / self.state_counts[i]
            else:
                self.transition_matrix[i] = 1.0 / self.n_numbers
                
    def get_transition_probabilities(self, current_state):
        """
        Retorna probabilidades de transição para um estado atual
        """
        return self.transition_matrix[current_state-1]

class CombinationAnalyzer:
    """
    Analisa padrões combinatórios nos números sorteados
    """
    def __init__(self, n_numbers=25, k_choose=15):
        self.n_numbers = n_numbers
        self.k_choose = k_choose
        self.pattern_counts = defaultdict(int)
        self.total_patterns = 0
        
    def analyze_patterns(self, sequences):
        """
        Analisa padrões nos números sorteados
        """
        for sequence in sequences:
            # Análise de pares adjacentes
            pairs = list(zip(sequence[:-1], sequence[1:]))
            for pair in pairs:
                self.pattern_counts[f"pair_{pair}"] += 1
            
            # Análise de gaps entre números
            gaps = np.diff(sequence)
            for gap in gaps:
                self.pattern_counts[f"gap_{gap}"] += 1
            
            # Análise de distribuição de paridade
            even_count = sum(1 for num in sequence if num % 2 == 0)
            self.pattern_counts[f"even_{even_count}"] += 1
            
            self.total_patterns += 1
    
    def get_pattern_probabilities(self):
        """
        Retorna probabilidades dos padrões identificados
        """
        probabilities = {}
        for pattern, count in self.pattern_counts.items():
            probabilities[pattern] = count / self.total_patterns
        return probabilities

class TimeSeriesAnalyzer:
    """
    Realiza análises de séries temporais nos dados
    """
    def __init__(self):
        self.arima_models = {}
        self.seasonal_patterns = {}
        
    def fit_arima(self, data, number):
        """
        Ajusta modelo ARIMA para um número específico
        """
        try:
            model = ARIMA(data, order=(5,1,2))
            self.arima_models[number] = model.fit()
            return True
        except:
            return False
            
    def analyze_seasonality(self, sequences, window_size=30):
        """
        Analisa padrões sazonais nos dados
        """
        number_occurrences = defaultdict(list)
        
        for i, sequence in enumerate(sequences):
            for number in sequence:
                number_occurrences[number].append(i)
        
        for number, occurrences in number_occurrences.items():
            if len(occurrences) > window_size:
                diffs = np.diff(occurrences)
                self.seasonal_patterns[number] = np.mean(diffs)

class PatternLearner:
    """
    Implementa aprendizado de padrões complexos usando técnicas avançadas
    """
    def __init__(self, sequence_length=10, n_numbers=25):
        self.sequence_length = sequence_length
        self.n_numbers = n_numbers
        self.pattern_memory = defaultdict(list)
        self.frequency_matrix = np.zeros((n_numbers, n_numbers))
        self.success_patterns = defaultdict(int)
        
    def update_frequency_matrix(self, sequences):
        """
        Atualiza matriz de frequência de co-ocorrência de números
        """
        for sequence in sequences:
            for i in range(len(sequence)):
                for j in range(i + 1, len(sequence)):
                    num1, num2 = sequence[i] - 1, sequence[j] - 1
                    self.frequency_matrix[num1][num2] += 1
                    self.frequency_matrix[num2][num1] += 1
        
        # Normaliza a matriz
        row_sums = self.frequency_matrix.sum(axis=1)
        self.frequency_matrix = self.frequency_matrix / row_sums[:, np.newaxis]
        
    def find_winning_patterns(self, historical_data, predictions):
        """
        Identifica padrões que levaram a previsões bem-sucedidas
        """
        for pred, actual in zip(predictions, historical_data):
            matches = len(set(pred) & set(actual))
            if matches >= 11:  # Padrão considerado bem-sucedido se acertar 11 ou mais
                pattern = self._extract_pattern(pred)
                self.success_patterns[pattern] += 1
    
def _extract_pattern(self, numbers):
    """
    Extrai padrões característicos de uma sequência de números
    """
    pattern = {
        'gaps': [int(x) for x in np.diff(sorted(numbers)).tolist()],  # Convertendo para int
        'even_ratio': float(sum(1 for n in numbers if n % 2 == 0) / len(numbers)),  # Garantindo float
        'sum': int(sum(numbers)),  # Convertendo para int
        'variance': float(np.var(numbers))  # Convertendo para float
    }
    return json.dumps(pattern)
    
    def get_pattern_scores(self, candidate_numbers):
        """
        Calcula pontuação de um conjunto de números baseado em padrões bem-sucedidos
        """
        pattern = self._extract_pattern(candidate_numbers)
        base_score = self.success_patterns.get(pattern, 0)
        
        # Adiciona análise de co-ocorrência
        cooc_score = 0
        for i in range(len(candidate_numbers)):
            for j in range(i + 1, len(candidate_numbers)):
                num1, num2 = candidate_numbers[i] - 1, candidate_numbers[j] - 1
                cooc_score += self.frequency_matrix[num1][num2]
        
        return base_score + (cooc_score / (len(candidate_numbers) * (len(candidate_numbers) - 1) / 2))

class StochasticOptimizer:
    """
    Implementa otimização estocástica para seleção de números
    """
    def __init__(self, n_numbers=25, k_choose=15):
        self.n_numbers = n_numbers
        self.k_choose = k_choose
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.min_temperature = 0.01
        
    def optimize_selection(self, probabilities, pattern_learner, iterations=1000):
        """
        Usa simulated annealing para otimizar seleção de números
        """
        current_solution = self._initial_solution(probabilities)
        current_score = self._evaluate_solution(current_solution, pattern_learner)
        best_solution = current_solution.copy()
        best_score = current_score
        
        temp = self.temperature
        while temp > self.min_temperature and iterations > 0:
            neighbor = self._get_neighbor(current_solution)
            neighbor_score = self._evaluate_solution(neighbor, pattern_learner)
            
            delta = neighbor_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current_solution = neighbor
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_solution = current_solution.copy()
                    best_score = current_score
            
            temp *= self.cooling_rate
            iterations -= 1
        
        return sorted(best_solution)
    
    def _initial_solution(self, probabilities):
        """
        Gera solução inicial baseada em probabilidades
        """
        return np.random.choice(
            np.arange(1, self.n_numbers + 1),
            size=self.k_choose,
            replace=False,
            p=probabilities/np.sum(probabilities)
        )
    
    def _get_neighbor(self, solution):
        """
        Gera solução vizinha trocando um número
        """
        neighbor = solution.copy()
        idx = np.random.randint(len(neighbor))
        available = list(set(range(1, self.n_numbers + 1)) - set(neighbor))
        neighbor[idx] = np.random.choice(available)
        return neighbor
    
    def _evaluate_solution(self, solution, pattern_learner):
        """
        Avalia qualidade da solução usando múltiplos critérios
        """
        pattern_score = pattern_learner.get_pattern_scores(solution)
        distribution_score = self._evaluate_distribution(solution)
        return pattern_score + distribution_score
    
    def _evaluate_distribution(self, solution):
        """
        Avalia distribuição dos números na solução
        """
        even_count = sum(1 for n in solution if n % 2 == 0)
        even_ratio = even_count / len(solution)
        optimal_ratio = 0.5  # Proporção ideal entre pares e ímpares
        distribution_score = 1 - abs(even_ratio - optimal_ratio)
        
        # Avalia gaps entre números consecutivos
        gaps = np.diff(sorted(solution))
        gap_variance = np.var(gaps)
        gap_score = 1 / (1 + gap_variance)  # Penaliza variância alta nos gaps
        
        return (distribution_score + gap_score) / 2

# No início do script, remova a configuração inicial do logging e adicione:
import logging
logger = logging.getLogger(__name__)

class LotteryPredictor:
    def __init__(self):
        self.data_file = 'dados.txt'
        self.models_dir = '.models/'
        self.data_dir = '.data/'
        self.sequence_length = 10
        self.num_numbers = 25
        self.numbers_to_choose = 15
        self.last_prediction = None
        self.last_prediction_date = None
        
        # Inicialização dos analisadores
        self.markov_chain = MarkovChain(self.num_numbers)
        self.combination_analyzer = CombinationAnalyzer()
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.pattern_learner = PatternLearner()
        self.stochastic_optimizer = StochasticOptimizer()
        
        # Primeiro, cria os diretórios necessários
        for directory in [self.models_dir, self.data_dir]:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    if os.name == 'nt':  # Se for Windows
                        os.system(f'attrib +h "{directory}"')
                except Exception as e:
                    print(f"Erro ao criar diretório {directory}: {e}")
        
        # Agora configura o logging após garantir que o diretório existe
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.data_dir, 'lottery_predictor.log')),
                logging.StreamHandler()
            ]
        )
        
        # Carregamento de dados
        self.historical_data = self._load_data('historical.pkl')
        self.predictions_history = self._load_data('predictions.pkl')
        self.performance_metrics = self._load_data('performance.pkl')
        
        if self.historical_data is None:  # Mudado de .empty para None
            logger.warning("Nenhum dado histórico encontrado. Inicializando novo DataFrame.")
            self.historical_data = pd.DataFrame()
        
        # Inicializa métricas de performance se necessário
        if self.performance_metrics is None:
            self.performance_metrics = {
                'accuracy_history': [],
                'pattern_success_rate': [],
                'prediction_hits': defaultdict(int)
            }

    def _load_data(self, filename):
        """Carrega dados do arquivo"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            return joblib.load(filepath)
        return None

    def _save_data(self, data, filename):
        """Salva dados em arquivo"""
        try:
            filepath = os.path.join(self.data_dir, filename)
            joblib.dump(data, filepath)
        except Exception as e:
            logger.error(f"Erro ao salvar {filename}: {e}")

    def process_new_data(self):
        """Processa novos dados e atualiza modelos"""
        if not os.path.exists(self.data_file) or os.path.getsize(self.data_file) == 0:
            return False

        try:
            new_data = pd.read_csv(self.data_file, sep='\t')
            if not new_data.empty:
                new_data['Data Sorteio'] = pd.to_datetime(new_data['Data Sorteio'], format='%d/%m/%Y')
                
                if not self.historical_data.empty:
                    new_data = new_data[new_data['Concurso'] > self.historical_data['Concurso'].max()]
                
                if not new_data.empty:
                    # Atualiza dados históricos
                    self.historical_data = pd.concat([self.historical_data, new_data])
                    self._save_data(self.historical_data, 'historical.pkl')
                    
                    # Avalia última previsão se existir
                    if self.last_prediction is not None:
                        self._evaluate_last_prediction(new_data.iloc[-1])
                    
                    # Atualiza todos os modelos
                    self._update_models()
                    
                    # Limpa arquivo de dados
                    open(self.data_file, 'w').close()
                    return True
            return False
        except Exception as e:
            logger.error(f"Erro ao processar novos dados: {e}")
            return False

    def _evaluate_last_prediction(self, actual_result):
        """Avalia a última previsão feita"""
        if self.last_prediction is None:
            return
        
        actual_numbers = [actual_result[f'Bola{i}'] for i in range(1, 16)]
        hits = len(set(self.last_prediction) & set(actual_numbers))
        
        # Atualiza métricas de performance
        self.performance_metrics['accuracy_history'].append(hits)
        self.performance_metrics['prediction_hits'][hits] += 1
        
        # Análise de padrões bem-sucedidos
        if hits >= 11:
            pattern = self.pattern_learner._extract_pattern(self.last_prediction)
            self.pattern_learner.success_patterns[pattern] += 1
        
        self._save_data(self.performance_metrics, 'performance.pkl')
        
        logger.info(f"Última previsão obteve {hits} acertos")

    def _update_models(self):
        """Atualiza todos os modelos com novos dados"""
        sequences = self._get_number_sequences()
        
        # Atualiza Cadeia de Markov
        self.markov_chain.fit(sequences)
        
        # Atualiza análise combinatória
        self.combination_analyzer.analyze_patterns(sequences)
        
        # Atualiza análise de séries temporais
        for num in range(1, self.num_numbers + 1):
            number_history = self._get_number_history(num)
            self.time_series_analyzer.fit_arima(number_history, num)
        
        # Atualiza aprendizado de padrões
        self.pattern_learner.update_frequency_matrix(sequences)

    def _get_number_sequences(self):
        """Obtém sequências de números dos dados históricos"""
        sequences = []
        for _, row in self.historical_data.iterrows():
            sequence = sorted([int(row[f'Bola{i}']) for i in range(1, 16)])
            sequences.append(sequence)
        return sequences

    def _get_number_history(self, number):
        """Obtém histórico de aparições de um número específico"""
        history = []
        for _, row in self.historical_data.iterrows():
            appeared = 1 if number in [row[f'Bola{i}'] for i in range(1, 16)] else 0
            history.append(appeared)
        return np.array(history)

    def _calculate_base_probabilities(self):
    """Calcula probabilidades base para cada número usando múltiplas análises"""
    probabilities = np.zeros(self.num_numbers)
    
    # 1. Probabilidades da Cadeia de Markov
    if self.historical_data.empty:
        return np.ones(self.num_numbers) / self.num_numbers
        
    last_numbers = sorted([int(self.historical_data.iloc[-1][f'Bola{i}']) for i in range(1, 16)])
    markov_probs = np.zeros(self.num_numbers)
    for last_num in last_numbers:
        markov_probs += self.markov_chain.get_transition_probabilities(last_num)
    
    # Verifica se há números válidos antes de dividir
    if len(last_numbers) > 0:
        markov_probs /= len(last_numbers)
    else:
        markov_probs = np.ones(self.num_numbers) / self.num_numbers
    
    # 2. Probabilidades baseadas em padrões
    pattern_probs = np.zeros(self.num_numbers)
    pattern_data = self.combination_analyzer.get_pattern_probabilities()
    for num in range(1, self.num_numbers + 1):
        pattern_probs[num-1] = sum(v for k, v in pattern_data.items() if f"_{num}" in k)
    
    # Verifica se a soma é zero antes de normalizar
    pattern_sum = np.sum(pattern_probs)
    if pattern_sum > 0:
        pattern_probs = pattern_probs / pattern_sum
    else:
        pattern_probs = np.ones(self.num_numbers) / self.num_numbers
    
    # 3. Probabilidades baseadas em séries temporais
    arima_probs = np.zeros(self.num_numbers)
    for num in range(1, self.num_numbers + 1):
        if num in self.time_series_analyzer.arima_models:
            try:
                forecast = self.time_series_analyzer.arima_models[num].forecast(1)
                arima_probs[num-1] = max(0, min(1, forecast[0]))
            except:
                arima_probs[num-1] = 1/self.num_numbers
    
    # Verifica se a soma é zero antes de normalizar
    arima_sum = np.sum(arima_probs)
    if arima_sum > 0:
        arima_probs = arima_probs / arima_sum
    else:
        arima_probs = np.ones(self.num_numbers) / self.num_numbers
    
    # 4. Análise de Recência
    recency_probs = np.zeros(self.num_numbers)
    for num in range(1, self.num_numbers + 1):
        history = self._get_number_history(num)
        if len(history) > 0:
            last_appearance = len(history) - 1 - np.argmax(history[::-1])
            recency_probs[num-1] = 1 / (1 + last_appearance)
        else:
            recency_probs[num-1] = 1 / self.num_numbers
    
    # Verifica se a soma é zero antes de normalizar
    recency_sum = np.sum(recency_probs)
    if recency_sum > 0:
        recency_probs = recency_probs / recency_sum
    else:
        recency_probs = np.ones(self.num_numbers) / self.num_numbers
    
    # Combina todas as probabilidades com pesos diferentes
    weights = [0.3, 0.25, 0.25, 0.2]  # Pesos para cada tipo de probabilidade
    probabilities = (weights[0] * markov_probs +
                    weights[1] * pattern_probs +
                    weights[2] * arima_probs +
                    weights[3] * recency_probs)
    
    # Verifica se há NaN e substitui por distribuição uniforme se necessário
    if np.any(np.isnan(probabilities)):
        logger.warning("Detectados valores NaN nas probabilidades. Usando distribuição uniforme.")
        return np.ones(self.num_numbers) / self.num_numbers
    
    # Garante que a soma das probabilidades é 1
    probabilities = probabilities / np.sum(probabilities)
    
    return probabilities

    def generate_prediction(self):
        """Gera nova previsão utilizando todos os modelos e análises"""
        try:
            if self.historical_data.empty:
                logger.error("Não há dados históricos suficientes para gerar previsão")
                return None
            
            # Calcula probabilidades base
            base_probabilities = self._calculate_base_probabilities()
            
            # Usa otimização estocástica para gerar conjunto final de números
            optimized_numbers = self.stochastic_optimizer.optimize_selection(
                base_probabilities,
                self.pattern_learner
            )
            
            # Valida a previsão
            if not self._validate_prediction(optimized_numbers):
                logger.warning("Previsão gerada não passou na validação. Gerando nova previsão...")
                return self.generate_prediction()
            
            # Atualiza última previsão
            self.last_prediction = optimized_numbers
            self.last_prediction_date = datetime.now()
            
            # Salva previsão no histórico
            prediction_record = {
                'data': self.last_prediction_date,
                'numeros': optimized_numbers,
                'concurso': self.historical_data['Concurso'].max() + 1
            }
            
            if self.predictions_history is None:
                self.predictions_history = pd.DataFrame([prediction_record])
            else:
                self.predictions_history = pd.concat([
                    self.predictions_history,
                    pd.DataFrame([prediction_record])
                ])
            
            self._save_data(self.predictions_history, 'predictions.pkl')
            
            return optimized_numbers
            
        except Exception as e:
            logger.error(f"Erro na geração de previsão: {e}")
            traceback.print_exc()
            return None

    def _validate_prediction(self, numbers):
        """Valida a previsão gerada"""
        if len(numbers) != self.numbers_to_choose:
            return False
            
        if len(set(numbers)) != self.numbers_to_choose:
            return False
            
        if not all(1 <= n <= self.num_numbers for n in numbers):
            return False
            
        # Verifica se não é igual à última previsão
        if self.last_prediction is not None:
            if set(numbers) == set(self.last_prediction):
                return False
                
        # Verifica se não é igual a nenhum resultado histórico
        for _, row in self.historical_data.iterrows():
            historical_numbers = set(int(row[f'Bola{i}']) for i in range(1, 16))
            if set(numbers) == historical_numbers:
                return False
                
        return True 

    def generate_performance_report(self):
        """Gera relatório detalhado de performance do modelo"""
        if not self.performance_metrics['accuracy_history']:
            return "Sem dados de performance disponíveis ainda."
        
        report = []
        report.append("\n=== RELATÓRIO DE PERFORMANCE ===")
        
        # Estatísticas gerais
        total_predictions = len(self.performance_metrics['accuracy_history'])
        avg_hits = np.mean(self.performance_metrics['accuracy_history'])
        report.append(f"\nTotal de previsões avaliadas: {total_predictions}")
        report.append(f"Média de acertos: {avg_hits:.2f}")
        
        # Distribuição de acertos
        report.append("\nDistribuição de acertos:")
        for hits, count in sorted(self.performance_metrics['prediction_hits'].items()):
            report.append(f"{hits} acertos: {count} vezes ({(count/total_predictions)*100:.1f}%)")
        
        return "\n".join(report)

    def analyze_current_trends(self):
        """Analisa tendências atuais nos dados"""
        if self.historical_data.empty:
            return "Sem dados históricos para análise."
        
        trends = []
        trends.append("\n=== ANÁLISE DE TENDÊNCIAS ATUAIS ===")
        
        # Análise dos últimos 30 sorteios
        recent_data = self.historical_data.tail(30)
        number_freq = defaultdict(int)
        
        for _, row in recent_data.iterrows():
            for i in range(1, 16):
                number_freq[int(row[f'Bola{i}'])] += 1
        
        # Números mais frequentes
        trends.append("\nNúmeros mais frequentes (últimos 30 sorteios):")
        most_frequent = sorted(number_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        for num, freq in most_frequent:
            trend = self._calculate_number_trend(num)
            trend_symbol = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            trends.append(f"Número {num}: {freq} vezes {trend_symbol}")
        
        # Números menos frequentes
        trends.append("\nNúmeros menos frequentes (últimos 30 sorteios):")
        least_frequent = sorted(number_freq.items(), key=lambda x: x[1])[:5]
        for num, freq in least_frequent:
            trend = self._calculate_number_trend(num)
            trend_symbol = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            trends.append(f"Número {num}: {freq} vezes {trend_symbol}")
        
        # Análise de padrões
        trends.append("\nPadrões identificados:")
        patterns = self._analyze_recent_patterns()
        for pattern, description in patterns.items():
            trends.append(f"{pattern}: {description}")
        
        return "\n".join(trends)

    def _calculate_number_trend(self, number, window=10):
        """Calcula tendência recente de um número"""
        recent_history = self._get_number_history(number)[-window:]
        if len(recent_history) < window:
            return 0
        slope = np.polyfit(range(len(recent_history)), recent_history, 1)[0]
        return slope

    def _analyze_recent_patterns(self):
        """Analisa padrões recentes nos sorteios"""
        patterns = {}
        recent_draws = self._get_number_sequences()[-5:]  # Últimos 5 sorteios
        
        # Análise de paridade
        even_counts = [sum(1 for n in draw if n % 2 == 0) for draw in recent_draws]
        avg_even = np.mean(even_counts)
        patterns['Paridade'] = f"Média de números pares: {avg_even:.1f}"
        
        # Análise de soma
        sums = [sum(draw) for draw in recent_draws]
        avg_sum = np.mean(sums)
        patterns['Soma'] = f"Média da soma: {avg_sum:.1f}"
        
        # Análise de gaps
        gaps = [np.mean(np.diff(draw)) for draw in recent_draws]
        avg_gap = np.mean(gaps)
        patterns['Intervalos'] = f"Média de intervalo entre números: {avg_gap:.1f}"
        
        return patterns

    def run(self):
        """Executa o sistema de previsão"""
        logger.info("Iniciando sistema de previsão da Lotofácil...")
        
        # Processa novos dados
        if self.process_new_data():
            logger.info("Novos dados processados com sucesso.")
        
        # Gera previsão
        logger.info("Gerando nova previsão...")
        numbers = self.generate_prediction()
        
        if numbers is not None:
            print("\n=== PREVISÃO LOTOFÁCIL ===")
            print(f"\nNúmeros sugeridos para o próximo sorteio:")
            print(sorted(numbers))
            
            # Exibe análise de tendências
            print(self.analyze_current_trends())
            
            # Exibe relatório de performance
            print(self.generate_performance_report())
            
            # Exibe informações adicionais
            print("\nInformações adicionais:")
            print(f"Data da previsão: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            print(f"Concurso previsto: {self.historical_data['Concurso'].max() + 1}")
            
            # Salva logs detalhados
            logger.info(f"Previsão gerada com sucesso: {numbers}")
        else:
            print("Não foi possível gerar uma previsão.")
            logger.error("Falha na geração da previsão")

def main():
    predictor = LotteryPredictor()
    predictor.run()

if __name__ == "__main__":
    main()                                       