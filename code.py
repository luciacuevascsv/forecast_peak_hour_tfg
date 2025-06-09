# librerias

import argparse
import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, SGDRegressor, SGDClassifier, Ridge
import gc

from playsound import playsound

warnings.filterwarnings("ignore")

# comando

parser = argparse.ArgumentParser(description='Modelo de predicción de hora pico.')
parser.add_argument('--pruebas', nargs='+', type=int, required=True, help='Lista de pruebas exitosas')
parser.add_argument('--prueba', required=True)
parser.add_argument('--tipo', required=True)

args = parser.parse_args()
pruebas = args.pruebas
prueba = args.prueba
tipo = args.tipo

if tipo=="clasificacion":

    modelos = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'SGDClassifier': SGDClassifier(loss='log_loss', max_iter=1000, random_state=42),
        'MLP': MLPClassifier(max_iter=200, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=42)
    }

elif tipo=='regresion_demanda':

    modelos = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'Ridge': Ridge(),
        'SGDRegressor': SGDRegressor(max_iter=1000, random_state=42),
        'MLPRegressor': MLPRegressor(max_iter=200, random_state=42)
        }
    
elif tipo=='regresion_simple':

    modelos = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'Ridge': Ridge(),
        'SGDRegressor': SGDRegressor(max_iter=1000, random_state=42),
        'MLPRegressor': MLPRegressor(max_iter=500, random_state=42)
        }

# datos y resultados

notebook_dir = os.getcwd()

sound_path = os.path.abspath(os.path.join(notebook_dir, 'applause.wav')) 

data_path = os.path.abspath(os.path.join(notebook_dir, 'data.csv')) 
data_base = pd.read_csv(data_path)
data_base['fecha'] = pd.to_datetime(data_base['fecha']).dt.date

results_path = os.path.abspath(os.path.join(notebook_dir, 'results.csv')) 
if os.path.exists(results_path):
    resultados_previos = pd.read_csv(results_path)
else:
    resultados_previos = pd.DataFrame(columns=['region', 'season', 'modelo', 'prueba', 'tipo_modelo', 'accuracy', 'precision', 'recall', 'f1_score', 'top2_accuracy','mse', 'rmse', 'mae', 'acc_cercania', 'score_hora', 'rho', 'error_medio_horas', 'tiempo_entrenamiento', 'n_samples', 'inputs', 'pruebas', 'SMOTE_aplicado'])

# valores

regiones = data_base['region'].unique().tolist()
estaciones = data_base['season'].unique().tolist()
estaciones.append("Aseasonal")
modelos_escalado = ['MLP', 'MLPRegressor', 'Ridge', 'LogisticRegression', 'SGDClassifier', 'SGDRegressor']

# codigo

for region in regiones:
    print(f"Entrando en región {region}")
    data_base_region=data_base.copy()
    data_base_region = data_base_region[data_base_region['region'] == region]
    data_base_region = data_base_region.drop(columns=['region'])

    for estacion in estaciones:
        print(f"Entrando en región {region} - estacion {estacion}")

        data=data_base_region.copy()
        if estacion != 'Aseasonal':
            data = data[data['season'] == estacion]
            data = data.drop(columns=['season'])
        
        grouped = data.groupby(['fecha'])
        features = []

        for (fecha), group in grouped:
            row = {
                    'fecha': fecha,
                    'mes': group['mes'].values[0],
                    'dia_semana': group['dia_semana'].values[0],
                    'festivo': group['festivo'].values[0],
                    'locmes': group['locmes'].values[0],
                }
            
            if tipo == 'clasificacion' or tipo == 'regresion_simple':
                row['hora_pico'] = group['hora_pico'].values[0]

            if estacion == 'Aseasonal':
                row['season'] = group['season'].values[0]

            if 8 in pruebas:
                row['hora_pico_x-1d'] = group['hora_pico_x-1d'].values[0]

            if 9 in pruebas:
                row['hora_pico_x-7d'] = group['hora_pico_x-7d'].values[0]

            if 10 in pruebas:
                row['hora_pico_x-2d'] = group['hora_pico_x-2d'].values[0]

            for i in range(24):
                hora_str = str(i)

                if tipo == 'regresion_demanda':
                    colname = f'demanda_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 3 in pruebas:
                    colname = f'temperatura_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 4 in pruebas:
                    colname = f'humedad_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 5 in pruebas:
                    colname = f'demanda_x-1d_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 6 in pruebas:
                    colname = f'demanda_x-7d_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 7 in pruebas:
                    colname = f'demanda_x-2d_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 11 in pruebas:
                    colname = f'tendencia_demanda_dia_anterior{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 12 in pruebas:
                    colname = f'residuo_demanda_dia_anterior{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 13 in pruebas:
                    colname = f'tendencia_demanda_dia_semana_pasada_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 14 in pruebas:
                    colname = f'residuo_demanda_dia_semana_pasada_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 15 in pruebas:
                    colname = f'temperatura_x-1d_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 16 in pruebas:
                    colname = f'humedad_x-1d_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 17 in pruebas:
                    colname = f'delta_temperatura_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

                if 18 in pruebas:
                    colname = f'delta_humedad_{hora_str}'
                    row[colname] = group[colname].values[0] if colname in group.columns else np.nan

            features.append(row)

        print("Dataframe creado!")
        subset = pd.DataFrame(features)


        for nombre_modelo, modelo in modelos.items():
                SMOTE_boolean=False
                ya_entrenado = (resultados_previos['tipo_modelo'] == tipo) & \
                                (resultados_previos['region'] == region) & \
                            (resultados_previos['season'] == estacion) & \
                            (resultados_previos['modelo'] == nombre_modelo) & \
                            (resultados_previos['prueba'] == prueba)
                if not resultados_previos[ya_entrenado].empty:
                    print(f"Saltando {region}-{estacion}-{nombre_modelo} (ya entrenado)")
                    continue

                print(f"Entrenando modelo {nombre_modelo} para {region}-{estacion}")

                try:
                    if tipo=='clasificacion' or tipo=='regresion_simple':

                        X = subset.drop(columns=['hora_pico', 'fecha'])
                        y = subset['hora_pico']

                    elif tipo=='regresion_demanda':

                        X = subset.drop(columns=['fecha'] + [f'demanda_{i}' for i in range(24)])
                        y = subset[[f'demanda_{i}' for i in range(24)]]
                        
                    cat_cols = ['locmes', 'mes', 'dia_semana'] + (['season'] if estacion == 'Aseasonal' else [])

                    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
                    X = X.apply(pd.to_numeric, errors='coerce').dropna()
                    y = y.loc[X.index]

                    subset = subset.loc[X.index]  

                    subset['fecha'] = pd.to_datetime(subset['fecha'].astype(str), errors='coerce')

                    train_mask = subset['fecha'].dt.year.between(2014, 2020)
                    test_mask = subset['fecha'].dt.year.between(2021, 2023)

                    X_train = X[train_mask]
                    y_train = y[train_mask]
                    X_test = X[test_mask]
                    y_test = y[test_mask]

                    if 2 in pruebas and tipo == 'clasificacion':
                        class_counts = y_train.value_counts()
                        clases_poco_representadas = class_counts[class_counts == 1].index.tolist()
                        if clases_poco_representadas:
                            print(f"⚠️ Eliminando clases con solo 1 instancia del entrenamiento: {clases_poco_representadas}")
                            
                            mask_train = ~y_train.isin(clases_poco_representadas)
                            X_train = X_train[mask_train]
                            y_train = y_train[mask_train]

                            mask_test = ~y_test.isin(clases_poco_representadas)
                            X_test = X_test[mask_test]
                            y_test = y_test[mask_test]

                        class_counts = Counter(y_train)
                        clases_distintas = len(class_counts)

                        if clases_distintas < 3:
                            print(f"⚠️ SMOTE no aplicado: solo hay {clases_distintas} clase(s) en el entrenamiento.")
                        else:
                            min_class_size = min(class_counts.values())
                            k_neighbors = min(5, min_class_size - 1)
                            
                            if k_neighbors >= 1:
                                try:
                                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                                    X_train, y_train = smote.fit_resample(X_train, y_train)
                                    print(f"✅ SMOTE aplicado con k_neighbors={k_neighbors}")
                                    SMOTE_boolean=True
                                except Exception as smote_error:
                                    print(f"⚠️ Error aplicando SMOTE: {smote_error}. Continuando sin SMOTE.")
                            else:
                                print(f"⚠️ SMOTE no aplicado: cada clase necesita al menos 2 ejemplos. Clase más pequeña tiene {min_class_size}")

                    if len(np.unique(y_train)) < 2 and tipo== 'clasificacion':
                        print(f"⚠️ No hay suficientes clases para entrenar en {region} - {estacion}. Saltando.")
                        continue

                    if tipo== 'clasificacion':
                        clases_train_unicas = np.unique(y_train)
                        test_mask_filtrado = np.isin(y_test, clases_train_unicas)

                        clases_test_originales = np.unique(y_test)
                        clases_faltantes = set(clases_test_originales) - set(clases_train_unicas)

                        if clases_faltantes:
                            print(f"⚠️ Clases en test no vistas en entrenamiento: {clases_faltantes}")

                        X_test = X_test[test_mask_filtrado]
                        y_test = y_test[test_mask_filtrado]

                        le = LabelEncoder()
                        y_all = pd.concat([y_train, y_test])
                        le.fit(y_all)

                        y_train = le.transform(y_train)
                        y_test = le.transform(y_test)
                        n_clases = len(le.classes_)

                    if nombre_modelo in modelos_escalado:
                        num_cols = [col for col in X_train.columns if any(keyword in col for keyword in ['demanda', 'temperatura', 'humedad'])]
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', StandardScaler(), num_cols),
                                ('cat', 'passthrough', [col for col in X.columns if col not in num_cols])
                            ]
                        )
                        modelo_final = Pipeline([
                            ('preproc', preprocessor),
                            ('clf', modelo)
                        ])
                    else:
                        modelo_final = modelo

                    if tipo== 'regresion_demanda':
                        modelo_final = MultiOutputRegressor(modelo_final)

                    print(f"🧠 Columnas usadas como input para {nombre_modelo} ({region} - {estacion}):")
                    print(list(X_train.columns))

                    print("Entrenando!")
                    start_time = time.time()
                    modelo_final.fit(X_train, y_train)
                    elapsed_time = time.time() - start_time
                    print("Entrenamiento completado!")

                    y_pred = modelo_final.predict(X_test)

                    if tipo== 'clasificacion':
                        y_proba = modelo_final.predict_proba(X_test) if hasattr(modelo_final, 'predict_proba') else None
                        if y_proba is not None:
                            y_proba_df = pd.DataFrame(y_proba, columns=modelo_final.classes_)
                            for cls in range(n_clases):
                                if cls not in y_proba_df.columns:
                                    y_proba_df[cls] = 0.0
                            y_proba_df = y_proba_df[sorted(y_proba_df.columns)]
                            top2 = top_k_accuracy_score(y_test, y_proba_df.values, k=2, labels=list(range(n_clases)))

                        else:
                            top2 = np.nan
                        
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        tolerancia_1h = (np.abs(y_pred - y_test) <= 1).mean()

                        y_test_decod = le.inverse_transform(y_test)
                        y_pred_decod = le.inverse_transform(y_pred)

                        errores = [abs(yt - yp) for yt, yp in zip(y_test_decod, y_pred_decod)]
                        max_error = 23 
                        puntuaciones = [1 - (err / max_error) for err in errores]
                        score_prop = sum(puntuaciones) / len(puntuaciones)

                        resultados_previos = pd.concat([resultados_previos, pd.DataFrame([{
                            'region': region,
                            'season': estacion,
                            'modelo': nombre_modelo,
                            'prueba': prueba,
                            'tipo_modelo': 'clasificacion',
                            'accuracy': acc,
                            'precision': prec,
                            'recall': rec,
                            'f1_score': f1,
                            'top2_accuracy': top2,
                            'acc_cercania': tolerancia_1h,
                            'score_hora': score_prop,
                            'tiempo_entrenamiento': elapsed_time,
                            'mse': None,
                            'rmse': None,
                            'mae': None,
                            'rho': None,
                            'error_medio_horas': None,
                            'n_samples': len(y_test),
                            'inputs': list(X_train.columns),
                            'pruebas': pruebas,
                            'SMOTE_aplicado': SMOTE_boolean

                        }])], ignore_index=True)

                        print(f"✅ {nombre_modelo}: accuracy={acc:.3f}, acc_cercania={tolerancia_1h:.3f}, top2={top2:.3f}, tiempo={elapsed_time:.2f}s")

                    if tipo== 'regresion_demanda':
                        hora_real = y_test.values.argmax(axis=1)
                        hora_predicha = y_pred.argmax(axis=1)
                        all_labels = np.unique(np.concatenate([hora_real, hora_predicha]))

                        acc = (hora_predicha == hora_real).mean()
                        tolerancia_1h = (np.abs(hora_predicha - hora_real) <= 1).mean()
                        rho, _ = spearmanr(hora_real, hora_predicha)
                        error_medio_horas = np.abs(hora_predicha - hora_real).mean()
                        f1 = f1_score(hora_real, hora_predicha, average='weighted', labels=all_labels)

                        errores = [abs(yt - yp) for yt, yp in zip(hora_real, hora_predicha)]
                        max_error = 23  
                        puntuaciones = [1 - (err / max_error) for err in errores]
                        score_prop = sum(puntuaciones) / len(puntuaciones)

                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)

                        resultados_previos = pd.concat([resultados_previos, pd.DataFrame([{
                            'region': region,
                            'season': estacion,
                            'modelo': nombre_modelo,
                            'prueba': prueba,
                            'tipo_modelo': 'regresion_demanda',
                            'accuracy': acc,
                            'mse': mse,
                            'rmse': rmse,
                            'mae': mae,
                            'acc_cercania': tolerancia_1h,
                            'rho': rho,
                            'score_hora': score_prop,
                            'error_medio_horas': error_medio_horas,
                            'tiempo_entrenamiento': elapsed_time,
                            'precision': None,
                            'recall': None,
                            'f1_score': f1,
                            'top2_accuracy': None,
                            'n_samples': len(y_test),
                            'inputs': list(X_train.columns),
                            'pruebas': pruebas,
                            'SMOTE_aplicado': SMOTE_boolean

                        }])], ignore_index=True)
                        print(f"✅ {nombre_modelo}: accuracy={acc:.3f}, acc_cercania={tolerancia_1h:.3f}, tiempo={elapsed_time:.2f}s")

                    if tipo== 'regresion_simple':
                        hora_predicha = np.rint(y_pred).astype(int)
                        hora_real = np.rint(y_test.values).astype(int)
                        all_labels = np.unique(np.concatenate([hora_real, hora_predicha]))

                        acc = (hora_predicha == hora_real).mean()
                        tolerancia_1h = (np.abs(hora_predicha - hora_real) <= 1).mean()
                        rho, _ = spearmanr(hora_real, hora_predicha)
                        error_medio_horas = np.abs(hora_predicha - hora_real).mean()
                        f1 = f1_score(hora_real, hora_predicha, average='weighted', labels=all_labels)

                        errores = [abs(yt - yp) for yt, yp in zip(hora_real, hora_predicha)]
                        max_error = 23 
                        puntuaciones = [1 - (err / max_error) for err in errores]
                        score_prop = sum(puntuaciones) / len(puntuaciones)

                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)

                        resultados_previos = pd.concat([resultados_previos, pd.DataFrame([{
                            'region': region,
                            'season': estacion,
                            'modelo': nombre_modelo,
                            'prueba': prueba,
                            'tipo_modelo': 'regresion_simple',
                            'accuracy': acc,
                            'mse': mse,
                            'rmse': rmse,
                            'mae': mae,
                            'acc_cercania': tolerancia_1h,
                            'rho': rho,
                            'score_hora': score_prop,
                            'error_medio_horas': error_medio_horas,
                            'tiempo_entrenamiento': elapsed_time,
                            'precision': None,
                            'recall': None,
                            'f1_score': f1,
                            'top2_accuracy': None,
                            'n_samples': len(y_test),
                            'inputs': list(X_train.columns),
                            'pruebas': pruebas,
                            'SMOTE_aplicado': SMOTE_boolean

                        }])], ignore_index=True)
                        print(f"✅ {nombre_modelo}: accuracy={acc:.3f}, acc_cercania={tolerancia_1h:.3f}, tiempo={elapsed_time:.2f}s")

                except Exception as e:
                    print(f"Error al entrenar {nombre_modelo} para {region}-{estacion}: {e}")

                resultados_previos.to_csv(results_path, index=False)
                print("Resultados guardados correctamente!")

                del modelo_final, X_train, y_train, X_test, y_test, X, y
                gc.collect()
                modelo_final = None

# sonido

def notificacion_sonora():
    playsound(sound_path)

notificacion_sonora()

                    








            
        

