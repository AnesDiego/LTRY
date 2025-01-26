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
            
        # Calcula recência (quantos sorteios desde última aparição)
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
        
        # Análise de pares consecutivos
        for idx in range(len(self.data)):
            numbers = sorted([self.data.iloc[idx][f'Bola{i}'] for i in range(1, 16)])
            for i in range(len(numbers)-1):
                pair = (numbers[i], numbers[i+1])
                patterns['consecutive_pairs'][pair] = patterns['consecutive_pairs'].get(pair, 0) + 1
        
        # Análise de posições mais frequentes
        for pos in range(1, 16):
            patterns['position_frequencies'][pos] = {}
            for num in range(1, self.num_numbers + 1):
                patterns['position_frequencies'][pos][num] = sum(self.data[f'Bola{pos}'] == num)
        
        return patterns

    def get_correlation_metrics(self):
        """Analisa correlações entre números"""
        correlations = np.zeros((self.num_numbers, self.num_numbers))
        number_matrix = np.zeros((len(self.data), self.num_numbers))
        
        # Criar matriz de ocorrências
        for idx, row in self.data.iterrows():
            numbers = [int(row[f'Bola{i}']) - 1 for i in range(1, 16)]
            number_matrix[idx, numbers] = 1
            
        # Calcular correlações
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
            
            # Calcular tendência (positiva ou negativa)
            trend = np.polyfit(range(len(appearances)), appearances, 1)[0]
            trends[num] = trend
            
        return trends

    def calculate_number_weights(self):
        """Calcula pesos para cada número baseado em todas as métricas"""
        weights = np.zeros(self.num_numbers)
        
        # Obter todas as métricas
        frequencies, recency = self.get_frequency_metrics()
        patterns = self.get_pattern_metrics()
        correlations = self.get_correlation_metrics()
        trends = self.get_trend_metrics()
        
        # Normalizar e combinar métricas
        for i in range(self.num_numbers):
            freq_weight = sum(frequencies[k][i+1] for k in frequencies.keys()) / len(frequencies)
            recency_weight = 1 / (recency[i+1] + 1)  # Menor recência = maior peso
            trend_weight = trends[i+1]
            correlation_weight = np.mean(correlations[i])
            
            # Combinar pesos com diferentes importâncias
            weights[i] = (
                0.3 * freq_weight +
                0.25 * recency_weight +
                0.25 * trend_weight +
                0.2 * correlation_weight
            )
        
        # Normalizar pesos finais
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        return weights