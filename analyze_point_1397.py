#!/usr/bin/env python3
"""
Script para analisar o ponto 1397 e identificar padrões problemáticos
"""

import json
import pandas as pd
import numpy as np

def analyze_point_1397():
    """Analisa o ponto 1397 e seus vizinhos para identificar padrões"""
    
    # Carrega os dados
    with open('trajectory_response.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    trajectory_points = data['trajectory']
    
    # Converte para DataFrame
    df = pd.DataFrame(trajectory_points)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Filtra pontos válidos
    df = df[df['isValid'] == True]
    
    print(f"Total de pontos válidos: {len(df)}")
    
    # Analisa o ponto 1397 e seus vizinhos (índice 1396 no DataFrame filtrado)
    target_idx = 1396  # 1397 - 1 (índice baseado em 0)
    
    if target_idx < len(df):
        print(f"\n=== ANÁLISE DO PONTO {target_idx + 1} (original 1397) ===")
        
        # Mostra 5 pontos antes e depois
        start_idx = max(0, target_idx - 5)
        end_idx = min(len(df), target_idx + 6)
        
        for i in range(start_idx, end_idx):
            point = df.iloc[i]
            marker = " >>> " if i == target_idx else "     "
            print(f"{marker}Ponto {i+1}: {point['timestamp']} | "
                  f"Lat: {point['latitude']:.6f} | Lon: {point['longitude']:.6f} | "
                  f"Speed: {point['speed']} | Source: {point['dataSource']}")
        
        # Calcula distâncias entre pontos consecutivos
        print(f"\n=== DISTÂNCIAS ENTRE PONTOS CONSECUTIVOS ===")
        
        def calculate_distance(lat1, lon1, lat2, lon2):
            """Calcula distância usando fórmula de Haversine"""
            R = 6371000  # Raio da Terra em metros
            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            delta_lat = np.radians(lat2 - lat1)
            delta_lon = np.radians(lon2 - lon1)
            
            a = (np.sin(delta_lat/2)**2 + 
                 np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            return R * c
        
        for i in range(start_idx, end_idx - 1):
            dist = calculate_distance(
                df.iloc[i]['latitude'], df.iloc[i]['longitude'],
                df.iloc[i+1]['latitude'], df.iloc[i+1]['longitude']
            )
            marker = " >>> " if i == target_idx else "     "
            print(f"{marker}Distância {i+1} → {i+2}: {dist:.1f}m")
        
        # Analisa padrões de timestamp
        print(f"\n=== ANÁLISE DE TIMESTAMPS ===")
        for i in range(start_idx, end_idx - 1):
            time_diff = (df.iloc[i+1]['timestamp'] - df.iloc[i]['timestamp']).total_seconds()
            marker = " >>> " if i == target_idx else "     "
            print(f"{marker}Intervalo {i+1} → {i+2}: {time_diff:.0f}s")
        
        # Analisa padrões de dataSource
        print(f"\n=== ANÁLISE DE DATA SOURCE ===")
        sources = df.iloc[start_idx:end_idx]['dataSource'].value_counts()
        print("Fontes de dados na região:")
        for source, count in sources.items():
            print(f"  {source}: {count} pontos")
        
        # Identifica possíveis problemas
        print(f"\n=== POSSÍVEIS PROBLEMAS IDENTIFICADOS ===")
        
        # 1. Verifica se há salto temporal grande
        if target_idx > 0:
            time_diff = (df.iloc[target_idx]['timestamp'] - df.iloc[target_idx-1]['timestamp']).total_seconds()
            if time_diff > 300:  # Mais de 5 minutos
                print(f"⚠️  Salto temporal grande: {time_diff:.0f}s")
        
        # 2. Verifica se há salto espacial grande
        if target_idx > 0:
            dist = calculate_distance(
                df.iloc[target_idx-1]['latitude'], df.iloc[target_idx-1]['longitude'],
                df.iloc[target_idx]['latitude'], df.iloc[target_idx]['longitude']
            )
            if dist > 100:  # Mais de 100m
                print(f"⚠️  Salto espacial grande: {dist:.1f}m")
        
        # 3. Verifica mudança de dataSource
        if target_idx > 0 and target_idx < len(df) - 1:
            prev_source = df.iloc[target_idx-1]['dataSource']
            curr_source = df.iloc[target_idx]['dataSource']
            next_source = df.iloc[target_idx+1]['dataSource']
            
            if prev_source != curr_source or curr_source != next_source:
                print(f"⚠️  Mudança de dataSource: {prev_source} → {curr_source} → {next_source}")
        
        # 4. Verifica se o ponto está fora da sequência temporal
        if target_idx > 0 and target_idx < len(df) - 1:
            prev_time = df.iloc[target_idx-1]['timestamp']
            curr_time = df.iloc[target_idx]['timestamp']
            next_time = df.iloc[target_idx+1]['timestamp']
            
            if not (prev_time <= curr_time <= next_time):
                print(f"⚠️  Ponto fora da sequência temporal!")
                print(f"    Anterior: {prev_time}")
                print(f"    Atual:    {curr_time}")
                print(f"    Próximo:  {next_time}")

if __name__ == "__main__":
    analyze_point_1397()
