# Processador de Trajetórias de Maquinários

Este projeto processa dados de trajetória de maquinários a partir de arquivos JSON e gera visualizações interativas e estáticas.

## Funcionalidades

- **Processamento de dados**: Carrega e processa dados de trajetória do arquivo `trajectory_response.json`
- **Ordenação cronológica**: Organiza os pontos de trajetória em ordem temporal
- **Visualização estática**: Gera gráficos com matplotlib mostrando a trajetória e velocidade
- **Mapa interativo**: Cria um mapa HTML interativo com folium
- **Análise de velocidade**: Categoriza pontos por velocidade (parado, baixa, média, alta)
- **Resumo estatístico**: Exibe estatísticas detalhadas da trajetória

## Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

Execute o script principal:
```bash
python plot_trajectory.py
```

## Arquivos Gerados

- `trajectory_plot.png`: Gráfico estático com a trajetória e velocidade
- `trajectory_map.html`: Mapa interativo que pode ser aberto no navegador

## Estrutura dos Dados

O script espera um arquivo JSON com a seguinte estrutura:
```json
{
  "deviceName": "DTT-02",
  "userId": "user_id",
  "period": {
    "start": "2025-07-08T00:00:00-03:00",
    "end": "2025-07-08T23:59:59-03:00"
  },
  "trajectory": [
    {
      "timestamp": "2025-07-08T23:58:32+00:00",
      "latitude": -18.08704,
      "longitude": -50.906664,
      "speed": "PARADO",
      "movement": "PARADO",
      "dataSource": "realtime",
      "isValid": true
    }
  ],
  "summary": {
    "totalPoints": 1974,
    "validPoints": 1974,
    "totalDistance": "55.89km",
    "timeSpan": "20h 57min",
    "averageSpeed": "4.1km/h",
    "maxSpeed": "141.1km/h"
  }
}
```

## Características do Mapa Interativo

- **Cores da trajetória**:
  - 🔴 Vermelho: Parado (velocidade = 0)
  - 🟠 Laranja: Baixa velocidade (< 20 km/h)
  - 🔵 Azul: Velocidade média (20-60 km/h)
  - 🟢 Verde: Alta velocidade (> 60 km/h)

- **Marcadores**:
  - 🟢 Início da trajetória
  - 🔴 Fim da trajetória
  - 🔴 Pontos de alta velocidade (> 80 km/h)

## Exemplo de Saída

O script exibe um resumo detalhado incluindo:
- Informações do dispositivo
- Período da trajetória
- Estatísticas de distância e velocidade
- Qualidade dos dados
- Análise de tempo parado vs. em movimento
