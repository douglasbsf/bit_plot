#!/usr/bin/env python3
"""
Script para processar e visualizar trajetórias de maquinários
Baseado nos dados do trajectory_response.json
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import folium
from datetime import datetime
import numpy as np
from typing import List, Dict, Any
import warnings
import os
from scipy import interpolate
from scipy.spatial.distance import cdist
warnings.filterwarnings('ignore')

class TrajectoryProcessor:
    """Classe para processar e visualizar trajetórias de maquinários"""
    
    def __init__(self, json_file_path: str = None):
        """
        Inicializa o processador com o arquivo JSON
        
        Args:
            json_file_path: Caminho para o arquivo trajectory_response.json (opcional)
        """
        self.json_file_path = json_file_path
        self.data = None
        self.df = None
        
    def load_data(self) -> None:
        """Carrega os dados do arquivo JSON"""
        print("Carregando dados do arquivo JSON...")
        with open(self.json_file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        print(f"Dados carregados: {self.data['summary']['totalPoints']} pontos")
        
    def process_trajectory(self) -> pd.DataFrame:
        """
        Processa a trajetória e ordena por timestamp
        
        Returns:
            DataFrame com os pontos ordenados cronologicamente
        """
        print("Processando trajetória...")
        
        # Extrai os pontos da trajetória
        trajectory_points = self.data['trajectory']
        
        # Converte para DataFrame
        df = pd.DataFrame(trajectory_points)
        
        # Converte timestamp para datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ordena por timestamp (cronológico)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Filtra apenas pontos válidos
        df = df[df['isValid'] == True]
        
        # Remove linhas com timestamps inválidos (NaT)
        df = df.dropna(subset=['timestamp'])
        
        # Converte velocidade para numérico quando possível
        df['speed_numeric'] = df['speed'].apply(self._parse_speed)
        
        # Aplica filtros de qualidade
        df = self._filter_outliers(df)
        df = self._smooth_trajectory(df)
        
        # Remove novamente linhas com timestamps inválidos após processamento
        df = df.dropna(subset=['timestamp'])
        
        self.df = df
        print(f"Trajetória processada: {len(df)} pontos válidos")
        print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
        
        return df
    
    def _parse_speed(self, speed_str: str) -> float:
        """
        Converte string de velocidade para valor numérico
        
        Args:
            speed_str: String da velocidade (ex: "PARADO", "45.2km/h")
            
        Returns:
            Valor numérico da velocidade em km/h
        """
        if speed_str == "PARADO":
            return 0.0
        
        try:
            # Remove "km/h" e converte para float
            speed_value = float(speed_str.replace("km/h", "").strip())
            return speed_value
        except:
            return 0.0
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calcula a distância entre dois pontos em metros usando fórmula de Haversine
        
        Args:
            lat1, lon1: Coordenadas do primeiro ponto
            lat2, lon2: Coordenadas do segundo ponto
            
        Returns:
            Distância em metros
        """
        R = 6371000  # Raio da Terra em metros
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _filter_outliers(self, df: pd.DataFrame, max_distance: float = 1000.0) -> pd.DataFrame:
        """
        Remove pontos outliers baseado na distância entre pontos consecutivos
        
        Args:
            df: DataFrame com os pontos da trajetória
            max_distance: Distância máxima em metros entre pontos consecutivos
            
        Returns:
            DataFrame filtrado
        """
        print("Filtrando outliers...")
        
        if len(df) < 3:
            return df
        
        # Calcula distâncias entre pontos consecutivos
        distances = []
        for i in range(1, len(df)):
            dist = self._calculate_distance(
                df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                df.iloc[i]['latitude'], df.iloc[i]['longitude']
            )
            distances.append(dist)
        
        # Identifica outliers
        outlier_indices = []
        for i, dist in enumerate(distances):
            if dist > max_distance:
                outlier_indices.append(i + 1)  # +1 porque distances começa do índice 1
        
        # Remove outliers
        df_filtered = df.drop(df.index[outlier_indices]).reset_index(drop=True)
        
        print(f"Removidos {len(outlier_indices)} pontos outliers")
        print(f"Pontos restantes: {len(df_filtered)}")
        
        return df_filtered
    
    def _smooth_trajectory(self, df: pd.DataFrame, max_distance_from_center: float = 3500.0) -> pd.DataFrame:
        """
        Suaviza a trajetória removendo apenas outliers muito óbvios e suavizando levemente
        
        Args:
            df: DataFrame com os pontos da trajetória
            max_distance_from_center: Distância máxima em metros do centro (muito alta para preservar dados)
            
        Returns:
            DataFrame com trajetória suavizada
        """
        print("Removendo apenas outliers muito óbvios e suavizando levemente...")
        
        if len(df) < 10:
            return df
        
        # Remove apenas outliers muito óbvios (fora da área de colheita)
        df_clean = self._remove_obvious_outliers(df, max_distance_from_center)
        
        # Aplica suavização muito leve em toda a trajetória
        df_smooth = df_clean.copy()
        
        # Suavização mais efetiva para linhas retas usando filtro gaussiano
        from scipy.ndimage import gaussian_filter1d
        
        if len(df_smooth) >= 10:
            # Suavização mais efetiva para tornar linhas retas realmente retas
            sigma = 2.0
            df_smooth['latitude'] = gaussian_filter1d(df_clean['latitude'].values, sigma=sigma)
            df_smooth['longitude'] = gaussian_filter1d(df_clean['longitude'].values, sigma=sigma)
        
        removed = len(df) - len(df_clean)
        print(f"Removidos apenas {removed} outliers óbvios")
        print(f"Pontos finais: {len(df_smooth)} (preservados {len(df_smooth)} pontos)")
        
        return df_smooth
    
    def _remove_obvious_outliers(self, df: pd.DataFrame, max_distance: float) -> pd.DataFrame:
        """
        Remove apenas outliers muito óbvios (fora da área de colheita)
        
        Args:
            df: DataFrame com os pontos da trajetória
            max_distance: Distância máxima em metros (muito alta para ser conservador)
            
        Returns:
            DataFrame sem outliers óbvios
        """
        if len(df) < 10:
            return df
        
        # Calcula a área principal de operação (centro dos dados)
        center_lat = df['latitude'].median()
        center_lon = df['longitude'].median()
        
        # Calcula distância de cada ponto ao centro
        distances = []
        for _, row in df.iterrows():
            dist = self._calculate_distance(
                center_lat, center_lon,
                row['latitude'], row['longitude']
            )
            distances.append(dist)
        
        # Remove apenas pontos muito distantes do centro (outliers óbvios)
        distances = np.array(distances)
        df_clean = df[distances <= max_distance].reset_index(drop=True)
        
        return df_clean
    
    def _remove_outliers_from_line(self, df: pd.DataFrame, max_distance: float) -> pd.DataFrame:
        """
        Remove pontos que estão muito distantes da linha reta principal
        
        Args:
            df: DataFrame com os pontos da trajetória
            max_distance: Distância máxima em metros de um ponto da linha reta
            
        Returns:
            DataFrame sem outliers
        """
        if len(df) < 5:
            return df
        
        # Calcula a linha reta principal (do primeiro ao último ponto)
        start_lat, start_lon = df.iloc[0]['latitude'], df.iloc[0]['longitude']
        end_lat, end_lon = df.iloc[-1]['latitude'], df.iloc[-1]['longitude']
        
        # Calcula distância de cada ponto à linha reta principal
        distances = []
        for _, row in df.iterrows():
            dist = self._distance_point_to_line(
                row['latitude'], row['longitude'],
                start_lat, start_lon, end_lat, end_lon
            )
            distances.append(dist)
        
        # Converte para array numpy e filtra pontos que estão muito distantes da linha
        distances = np.array(distances)
        df_clean = df[distances <= max_distance].reset_index(drop=True)
        
        return df_clean
    
    def _remove_outliers_in_segment(self, df_segment: pd.DataFrame, max_distance: float) -> pd.DataFrame:
        """
        Remove outliers dentro de um segmento específico (A→B→C, se B estiver fora da linha A-C)
        
        Args:
            df_segment: DataFrame com os pontos do segmento
            max_distance: Distância máxima em metros de um ponto da linha reta
            
        Returns:
            DataFrame do segmento sem outliers
        """
        if len(df_segment) < 3:
            return df_segment
        
        # Para cada ponto intermediário, verifica se está muito longe da linha
        valid_indices = [0]  # Sempre mantém o primeiro ponto
        
        for i in range(1, len(df_segment) - 1):
            # Calcula distância do ponto i à linha entre o ponto anterior e o próximo
            prev_idx = valid_indices[-1]
            next_idx = i + 1
            
            dist = self._distance_point_to_line(
                df_segment.iloc[i]['latitude'], df_segment.iloc[i]['longitude'],
                df_segment.iloc[prev_idx]['latitude'], df_segment.iloc[prev_idx]['longitude'],
                df_segment.iloc[next_idx]['latitude'], df_segment.iloc[next_idx]['longitude']
            )
            
            # Se a distância for aceitável, mantém o ponto
            if dist <= max_distance:
                valid_indices.append(i)
        
        # Sempre mantém o último ponto
        valid_indices.append(len(df_segment) - 1)
        
        return df_segment.iloc[valid_indices].reset_index(drop=True)
    
    def _distance_point_to_line(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calcula a distância de um ponto a uma linha reta
        
        Args:
            px, py: Coordenadas do ponto
            x1, y1: Coordenadas do primeiro ponto da linha
            x2, y2: Coordenadas do segundo ponto da linha
            
        Returns:
            Distância em metros
        """
        # Converte para metros (aproximação)
        px_m = px * 111000
        py_m = py * 111000 * np.cos(np.radians(px))
        x1_m = x1 * 111000
        y1_m = y1 * 111000 * np.cos(np.radians(x1))
        x2_m = x2 * 111000
        y2_m = y2 * 111000 * np.cos(np.radians(x2))
        
        # Fórmula da distância ponto-linha
        A = y2_m - y1_m
        B = x1_m - x2_m
        C = x2_m * y1_m - x1_m * y2_m
        
        distance = abs(A * px_m + B * py_m + C) / np.sqrt(A**2 + B**2)
        
        return distance
    
    def _identify_main_directions(self, df: pd.DataFrame, direction_threshold: float) -> List[tuple]:
        """
        Identifica direções principais baseado em mudanças de direção consistentes
        
        Args:
            df: DataFrame com os pontos da trajetória
            direction_threshold: Ângulo em graus para detectar mudanças de direção
            
        Returns:
            Lista de tuplas (start_idx, end_idx) para cada direção
        """
        if len(df) < 10:
            return [(0, len(df)-1)]
        
        directions = []
        start_idx = 0
        
        i = 5  # Começa analisando a partir do 5º ponto
        while i < len(df) - 5:
            # Analisa 5 pontos seguidos para confirmar mudança de direção
            direction_changed = True
            for j in range(5):
                if i + j < len(df) - 2:
                    angle = self._calculate_angle(
                        df.iloc[i+j-2]['latitude'], df.iloc[i+j-2]['longitude'],
                        df.iloc[i+j-1]['latitude'], df.iloc[i+j-1]['longitude'],
                        df.iloc[i+j]['latitude'], df.iloc[i+j]['longitude']
                    )
                    if angle <= direction_threshold:
                        direction_changed = False
                        break
            
            if direction_changed:
                # Confirma mudança de direção - adiciona a direção anterior
                directions.append((start_idx, i-1))
                start_idx = i-1
                i += 5  # Pula 5 pontos para evitar detecção dupla
            else:
                i += 1
        
        # Adiciona a última direção
        directions.append((start_idx, len(df)-1))
        
        return directions
    
    def _smooth_direction(self, values: np.ndarray) -> np.ndarray:
        """
        Suaviza valores dentro de uma direção usando filtro gaussiano mais agressivo
        
        Args:
            values: Array de valores para suavizar
            
        Returns:
            Array suavizado
        """
        if len(values) < 10:
            return values
        
        # Aplica filtro gaussiano mais agressivo
        from scipy.ndimage import gaussian_filter1d
        
        # Suavização mais agressiva para direções longas
        sigma = max(2.0, len(values) / 10.0)
        smoothed = gaussian_filter1d(values, sigma=sigma)
        
        return smoothed
    
    def _identify_straight_segments(self, df: pd.DataFrame, angle_threshold: float, min_segment_length: int) -> List[tuple]:
        """
        Identifica segmentos de linha reta baseado em mudanças de direção
        
        Args:
            df: DataFrame com os pontos da trajetória
            angle_threshold: Ângulo em graus para detectar mudanças de direção
            min_segment_length: Comprimento mínimo do segmento
            
        Returns:
            Lista de tuplas (start_idx, end_idx) para cada segmento
        """
        if len(df) < 3:
            return [(0, len(df)-1)]
        
        segments = []
        start_idx = 0
        
        for i in range(2, len(df)):
            # Calcula o ângulo entre três pontos consecutivos
            angle = self._calculate_angle(
                df.iloc[i-2]['latitude'], df.iloc[i-2]['longitude'],
                df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                df.iloc[i]['latitude'], df.iloc[i]['longitude']
            )
            
            # Se o ângulo for muito grande, é uma curva (mudança de direção)
            if angle > angle_threshold:
                # Só adiciona o segmento se for grande o suficiente
                if i-1 - start_idx >= min_segment_length:
                    segments.append((start_idx, i-1))
                start_idx = i-1
        
        # Adiciona o último segmento se for grande o suficiente
        if len(df)-1 - start_idx >= min_segment_length:
            segments.append((start_idx, len(df)-1))
        
        return segments
    
    def _calculate_angle(self, lat1: float, lon1: float, lat2: float, lon2: float, lat3: float, lon3: float) -> float:
        """
        Calcula o ângulo entre três pontos (em graus)
        
        Args:
            lat1, lon1: Primeiro ponto
            lat2, lon2: Ponto central
            lat3, lon3: Terceiro ponto
            
        Returns:
            Ângulo em graus
        """
        # Converte para vetores
        v1 = np.array([lat2 - lat1, lon2 - lon1])
        v2 = np.array([lat3 - lat2, lon3 - lon2])
        
        # Calcula o ângulo entre os vetores
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Evita erros numéricos
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _smooth_segment(self, values: np.ndarray) -> np.ndarray:
        """
        Suaviza valores dentro de um segmento usando filtro gaussiano
        
        Args:
            values: Array de valores para suavizar
            
        Returns:
            Array suavizado
        """
        if len(values) < 5:
            return values
        
        # Aplica filtro gaussiano para suavizar
        from scipy.ndimage import gaussian_filter1d
        
        # Suavização com sigma baseado no tamanho do segmento
        sigma = max(1.0, len(values) / 20.0)
        smoothed = gaussian_filter1d(values, sigma=sigma)
        
        return smoothed
    
    def generate_harvest_machine_simulation(self, 
                                          field_center_lat: float = -23.5505,
                                          field_center_lon: float = -46.6333,
                                          field_width: float = 0.01,  # ~1km
                                          field_length: float = 0.02,  # ~2km
                                          num_passes: int = 20,
                                          pass_width: float = 0.0005,  # ~50m
                                          speed_kmh: float = 8.0) -> None:
        """
        Gera uma simulação realista de uma máquina de colheita agrícola
        
        Args:
            field_center_lat: Latitude central do campo
            field_center_lon: Longitude central do campo
            field_width: Largura do campo em graus
            field_length: Comprimento do campo em graus
            num_passes: Número de passadas da máquina
            pass_width: Largura de cada passada em graus
            speed_kmh: Velocidade da máquina em km/h
        """
        print("Gerando simulação de máquina de colheita agrícola...")
        
        # Calcula os limites do campo
        lat_min = field_center_lat - field_length / 2
        lat_max = field_center_lat + field_length / 2
        lon_min = field_center_lon - field_width / 2
        lon_max = field_center_lon + field_width / 2
        
        trajectory_points = []
        current_time = datetime.now()
        
        # Gera as passadas da máquina (padrão de ida e volta)
        for pass_num in range(num_passes):
            # Calcula a posição Y da passada atual
            pass_lat = lat_min + (pass_num * pass_width)
            
            if pass_lat > lat_max:
                break
            
            # Determina a direção (ida ou volta)
            if pass_num % 2 == 0:
                # Ida: da esquerda para a direita
                start_lon = lon_min
                end_lon = lon_max
            else:
                # Volta: da direita para a esquerda
                start_lon = lon_max
                end_lon = lon_min
            
            # Gera pontos ao longo da passada
            num_points_per_pass = 50
            for i in range(num_points_per_pass):
                progress = i / (num_points_per_pass - 1)
                current_lon = start_lon + (end_lon - start_lon) * progress
                
                # Adiciona pequenas variações para simular movimento real
                lat_noise = np.random.normal(0, 0.00001)  # ~1m de variação
                lon_noise = np.random.normal(0, 0.00001)
                
                # Adiciona paradas ocasionais (manobras, descarga, etc.)
                if np.random.random() < 0.05:  # 5% chance de parar
                    current_speed = "PARADO"
                    # Fica parado por alguns pontos
                    for stop_point in range(3):
                        trajectory_points.append({
                            'latitude': pass_lat + lat_noise,
                            'longitude': current_lon + lon_noise,
                            'timestamp': current_time.strftime('%Y-%m-%dT%H:%M:%S'),
                            'speed': current_speed,
                            'isValid': True
                        })
                        current_time = current_time + pd.Timedelta(seconds=10)
                else:
                    # Velocidade normal com pequenas variações
                    speed_variation = np.random.normal(speed_kmh, speed_kmh * 0.1)
                    speed_variation = max(0, min(speed_variation, speed_kmh * 1.5))
                    current_speed = f"{speed_variation:.1f}km/h"
                    
                    trajectory_points.append({
                        'latitude': pass_lat + lat_noise,
                        'longitude': current_lon + lon_noise,
                        'timestamp': current_time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'speed': current_speed,
                        'isValid': True
                    })
                    current_time = current_time + pd.Timedelta(seconds=5)
        
        # Cria a estrutura de dados simulada
        self.data = {
            'deviceName': 'Colheitadeira John Deere S760',
            'userId': 'fazenda_simulada',
            'period': {
                'start': trajectory_points[0]['timestamp'],
                'end': trajectory_points[-1]['timestamp']
            },
            'trajectory': trajectory_points,
            'summary': {
                'totalPoints': len(trajectory_points),
                'validPoints': len(trajectory_points),
                'totalDistance': f"{len(trajectory_points) * 0.1:.1f} km",
                'timeSpan': f"{len(trajectory_points) * 5 / 3600:.1f} horas",
                'averageSpeed': f"{speed_kmh:.1f} km/h",
                'maxSpeed': f"{speed_kmh * 1.5:.1f} km/h"
            },
            'quality': {
                'dataHealth': 'Excelente'
            }
        }
        
        print(f"Simulação gerada: {len(trajectory_points)} pontos")
        print(f"Campo: {field_width*111:.0f}m x {field_length*111:.0f}m")
        print(f"Passadas: {num_passes}")
        print(f"Velocidade: {speed_kmh} km/h")
    
    def create_static_plot(self, save_path: str = "trajectory_plot.png") -> None:
        """
        Cria um gráfico estático da trajetória
        
        Args:
            save_path: Caminho para salvar o gráfico
        """
        print("Criando gráfico estático...")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Gráfico 1: Trajetória suavizada
        ax1.plot(self.df['longitude'], self.df['latitude'], 
                'b-', linewidth=2, alpha=0.8, label='Trajetória Suavizada')
        ax1.scatter(self.df['longitude'].iloc[0], self.df['latitude'].iloc[0], 
                   color='green', s=100, label='Início', zorder=5)
        ax1.scatter(self.df['longitude'].iloc[-1], self.df['latitude'].iloc[-1], 
                   color='red', s=100, label='Fim', zorder=5)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'Trajetória Suavizada do Maquinário {self.data["deviceName"]}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfico 2: Comparação - Original vs Suavizada
        # Carrega dados originais para comparação
        original_df = pd.DataFrame(self.data['trajectory'])
        original_df['timestamp'] = pd.to_datetime(original_df['timestamp'])
        original_df = original_df.sort_values('timestamp').reset_index(drop=True)
        original_df = original_df[original_df['isValid'] == True]
        
        ax2.plot(original_df['longitude'], original_df['latitude'], 
                'r-', linewidth=1, alpha=0.5, label='Trajetória Original (com ruído)')
        ax2.plot(self.df['longitude'], self.df['latitude'], 
                'b-', linewidth=2, alpha=0.8, label='Trajetória Suavizada')
        
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Comparação: Original vs Suavizada')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Gráfico 3: Velocidade ao longo do tempo
        ax3.plot(self.df['timestamp'], self.df['speed_numeric'], 
                'g-', linewidth=1, alpha=0.7)
        ax3.set_xlabel('Tempo')
        ax3.set_ylabel('Velocidade (km/h)')
        ax3.set_title('Velocidade ao Longo do Tempo')
        ax3.grid(True, alpha=0.3)
        
        # Rotaciona labels do eixo x para melhor legibilidade
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico salvo em: {save_path}")
    
    def create_interactive_map(self, save_path: str = "trajectory_map.html") -> None:
        """
        Cria um mapa interativo com a trajetória
        
        Args:
            save_path: Caminho para salvar o mapa HTML
        """
        print("Criando mapa interativo...")
        
        # Calcula o centro do mapa
        center_lat = self.df['latitude'].mean()
        center_lon = self.df['longitude'].mean()
        
        # Cria o mapa
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles='OpenStreetMap'
        )
        
        # Adiciona a trajetória como uma linha
        trajectory_coords = list(zip(self.df['latitude'], self.df['longitude']))
        
        # Cria uma linha colorida baseada na velocidade
        for i in range(len(trajectory_coords) - 1):
            speed = self.df['speed_numeric'].iloc[i]
            
            # Define cor baseada na velocidade
            if speed == 0:
                color = 'red'  # Parado
            elif speed < 20:
                color = 'orange'  # Baixa velocidade
            elif speed < 60:
                color = 'blue'  # Velocidade média
            else:
                color = 'green'  # Alta velocidade
            
            folium.PolyLine(
                [trajectory_coords[i], trajectory_coords[i + 1]],
                color=color,
                weight=3,
                opacity=0.7,
                popup=f"Velocidade: {speed:.1f} km/h"
            ).add_to(m)
        
        # Adiciona marcador de início
        folium.Marker(
            [self.df['latitude'].iloc[0], self.df['longitude'].iloc[0]],
            popup=f"Início<br>Hora: {self.df['timestamp'].iloc[0]}<br>Velocidade: {self.df['speed'].iloc[0]}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        # Adiciona marcador de fim
        folium.Marker(
            [self.df['latitude'].iloc[-1], self.df['longitude'].iloc[-1]],
            popup=f"Fim<br>Hora: {self.df['timestamp'].iloc[-1]}<br>Velocidade: {self.df['speed'].iloc[-1]}",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
        
        # Adiciona pontos de interesse (velocidade alta)
        high_speed_points = self.df[self.df['speed_numeric'] > 80]
        for _, point in high_speed_points.iterrows():
            folium.CircleMarker(
                [point['latitude'], point['longitude']],
                radius=5,
                popup=f"Alta Velocidade<br>Hora: {point['timestamp']}<br>Velocidade: {point['speed']}",
                color='red',
                fill=True,
                fillColor='red'
            ).add_to(m)
        
        # Adiciona legenda
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Legenda:</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> Parado</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Baixa Velocidade</p>
        <p><i class="fa fa-circle" style="color:blue"></i> Velocidade Média</p>
        <p><i class="fa fa-circle" style="color:green"></i> Alta Velocidade</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Salva o mapa
        m.save(save_path)
        print(f"Mapa interativo salvo em: {save_path}")
    
    def print_summary(self) -> None:
        """Imprime um resumo dos dados processados"""
        print("\n" + "="*50)
        print("RESUMO DA TRAJETÓRIA")
        print("="*50)
        print(f"Dispositivo: {self.data['deviceName']}")
        print(f"Usuário: {self.data['userId']}")
        print(f"Período: {self.data['period']['start']} a {self.data['period']['end']}")
        print(f"Total de pontos: {self.data['summary']['totalPoints']}")
        print(f"Pontos válidos: {self.data['summary']['validPoints']}")
        print(f"Distância total: {self.data['summary']['totalDistance']}")
        print(f"Tempo total: {self.data['summary']['timeSpan']}")
        print(f"Velocidade média: {self.data['summary']['averageSpeed']}")
        print(f"Velocidade máxima: {self.data['summary']['maxSpeed']}")
        print(f"Qualidade dos dados: {self.data['quality']['dataHealth']}")
        
        # Estatísticas adicionais
        print(f"\nEstatísticas dos dados processados:")
        print(f"Pontos processados: {len(self.df)}")
        print(f"Velocidade média calculada: {self.df['speed_numeric'].mean():.1f} km/h")
        print(f"Velocidade máxima calculada: {self.df['speed_numeric'].max():.1f} km/h")
        print(f"Tempo parado: {len(self.df[self.df['speed_numeric'] == 0])} pontos")
        print(f"Tempo em movimento: {len(self.df[self.df['speed_numeric'] > 0])} pontos")


def main():
    """Função principal"""
    print("Processador de Trajetórias de Maquinários Agrícolas")
    print("="*60)
    
    # Inicializa o processador com o arquivo JSON original
    processor = TrajectoryProcessor('trajectory_response.json')
    
    # Carrega e processa os dados originais
    processor.load_data()
    processor.process_trajectory()
    
    # Imprime resumo
    processor.print_summary()
    
    # Cria visualizações
    print("\nCriando visualizações...")
    processor.create_static_plot()
    processor.create_interactive_map()
    
    print("\nProcessamento concluído!")
    print("Arquivos gerados:")
    print("- trajectory_plot.png (gráfico estático)")
    print("- trajectory_map.html (mapa interativo)")
    print("\nTrajetória original preservada com suavização mínima")


if __name__ == "__main__":
    main()
