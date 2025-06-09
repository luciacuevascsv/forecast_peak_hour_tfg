import pandas as pd
import os
import numpy as np

notebook_dir = os.getcwd()
ruta_horario = os.path.abspath(os.path.join(notebook_dir, 'hourly.csv'))
ruta_csv = os.path.abspath(os.path.join(notebook_dir, 'baseline.csv'))
lock_path = ruta_csv + ".lock"

print("Cargando datos...")
horario = pd.read_csv(ruta_horario, parse_dates=["datetime"])
horario['hora'] = horario['hora'].astype(int)
horario['es_demanda_max'] = horario['es_demanda_max'].astype(bool)
print("Datos cargados correctamente.")

horario = horario.sort_values(['region', 'datetime'])

horario['fecha'] = horario['datetime'].dt.date
hora_pico = horario.loc[horario.groupby(['region', 'fecha'])['demanda'].idxmax()][['region', 'fecha', 'hora']]
hora_pico = hora_pico.rename(columns={'hora': 'hora_demanda_max'})

hora_pico = hora_pico.sort_values(['region', 'fecha'])
hora_pico['hora_predicha'] = hora_pico.groupby('region')['hora_demanda_max'].shift(1)

hora_pico['hora_predicha'] = hora_pico['hora_predicha'].fillna(hora_pico['hora_demanda_max'])

horario = horario.merge(hora_pico, on=['region', 'fecha'], how='left')

horario = horario.dropna(subset=['hora_predicha'])

horario['season_aseasonal'] = 'Aseasonal'

combinaciones = []
for region in horario['region'].unique():
    for season in horario['season'].unique():
        combinaciones.append((region, season))
    combinaciones.append((region, 'Aseasonal'))

resultados = []

for region, season in combinaciones:
    if season == 'Aseasonal':
        subset = horario[horario['region'] == region]
    else:
        subset = horario[(horario['region'] == region) & (horario['season'] == season)]

    subset_daily = subset.drop_duplicates(subset=['region', 'fecha'])

    if len(subset_daily) == 0:
        continue

    y_true = subset_daily['hora_demanda_max'].astype(int).values
    y_pred = subset_daily['hora_predicha'].astype(int).values
    n_clases = 24

    accuracy = (y_true == y_pred).mean()
    acc_cercania = (np.abs(y_true - y_pred) <= 1).mean()
    
    errores = [abs(yt - yp) for yt, yp in zip(y_true, y_pred)]
    max_error = n_clases - 1  
    puntuaciones = [1 - (err / max_error) for err in errores]
    score_prop = sum(puntuaciones) / len(puntuaciones)

    resultados.append({
        'region': region,
        'season': season,
        'accuracy': accuracy,
        'acc_cercania': acc_cercania,
        'score_hora': score_prop,
        'n_samples': len(subset_daily)
    })

resultados_df = pd.DataFrame(resultados)
resultados_df.to_csv(ruta_csv, index=False)

print('Modelo naive completado')
resultados_df.head()