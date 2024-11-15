import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, norm

# Cargar los datos desde el CSV
file_path = 'datos_procesados3.csv'
data = pd.read_csv(file_path)

# Crear los grupos basados en tus criterios
grupos_comparacion = {
    'Bebés <= 2500g': data['PESO_NAC'].isin([1, 2, 3, 4]),
    'Bebés > 2500g': data['PESO_NAC'].isin([5, 6, 7, 8, 9]),
    'Embarazos < 38 semanas': data['T_GES'].isin([1, 2, 3]),
    'Embarazos >= 38 semanas': data['T_GES'].isin([4, 5]),
    'Embarazos simples': data['MUL_PARTO'] == 1,
    'Embarazos múltiples': data['MUL_PARTO'].isin([2, 3, 4]),
    'Madres solteras': data['EST_CIVM'] == 5,
    'Madres comprometidas': data['EST_CIVM'].isin([1, 2]),
    'Madres con seguro contributivo': data['SEG_SOCIAL'] == 1,
    'Madres con seguro subsidiado': data['SEG_SOCIAL'] == 2,
    'Nacimientos urbanos': data['AREANAC'] == 1,
    'Nacimientos rurales': data['AREANAC'].isin([2, 3]),
    'Bebés masculinos': data['SEXO'] == 1,
    'Bebés femeninos': data['SEXO'] == 2
}

# Lista para almacenar los resultados
results_list = []

# Función para calcular Odds Ratio, p-value e intervalo de confianza
def calculate_odds_ratio_with_ci(data, group_condition, comparison_condition, group_name, comparison_name, year, alpha=0.05):
    a = data[group_condition & comparison_condition].shape[0]
    b = data[group_condition].shape[0] - a
    c = data[~group_condition & comparison_condition].shape[0]
    d = data[~group_condition].shape[0] - c

    # Verificar si la tabla tiene suficientes datos
    if a == 0 or b == 0 or c == 0 or d == 0:
        print(f"Año {year} - Advertencia: La tabla de contingencia entre '{group_name}' y '{comparison_name}' tiene valores cero. No se puede calcular el Odds Ratio.")
        results_list.append({
            'Año': year, 'Grupo 1': group_name, 'Grupo 2': comparison_name,
            'Odds Ratio': 'Valores cero', 'P-value': 'N/A',
            'CI Lower': 'N/A', 'CI Upper': 'N/A'
        })
        return

    # Crear tabla de contingencia
    contingency_table = np.array([[a, b], [c, d]])

    try:
        # Calcular el Odds Ratio
        odds_ratio = (a / b) / (c / d)

        # Calcular el intervalo de confianza
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        z = norm.ppf(1 - alpha/2)
        ci_lower = np.exp(np.log(odds_ratio) - z * se_log_or)
        ci_upper = np.exp(np.log(odds_ratio) + z * se_log_or)

        # Calcular el p-value usando fisher_exact de scipy
        _, p_value = fisher_exact(contingency_table)

        print(f"Año {year} - Odds Ratio entre '{group_name}' y '{comparison_name}':", odds_ratio)
        print(f"Año {year} - P-value entre '{group_name}' y '{comparison_name}':", p_value)
        print(f"Año {year} - Intervalo de confianza al 95% entre '{group_name}' y '{comparison_name}': [{ci_lower}, {ci_upper}]\n")

        # Agregar resultados a la lista
        results_list.append({
            'Año': year, 'Grupo 1': group_name, 'Grupo 2': comparison_name,
            'Odds Ratio': odds_ratio, 'P-value': p_value,
            'CI Lower': ci_lower, 'CI Upper': ci_upper
        })

    except ZeroDivisionError:
        print(f"Año {year} - Error: División por cero al calcular el Odds Ratio.\n")
        results_list.append({
            'Año': year, 'Grupo 1': group_name, 'Grupo 2': comparison_name,
            'Odds Ratio': 'División por cero', 'P-value': 'N/A',
            'CI Lower': 'N/A', 'CI Upper': 'N/A'
        })

# Iterar por cada año en la columna ANO
for year, group_data in data.groupby('ANO'):
    madres_menores_30 = group_data['EDAD_MADRE'].isin([1, 2, 3, 4])
    madres_mayores_30 = group_data['EDAD_MADRE'].isin([5, 6, 7, 8, 9])

    # Comparar madres menores de 30 con todos los grupos de comparación
    for group_name, condition in grupos_comparacion.items():
        calculate_odds_ratio_with_ci(group_data, madres_menores_30, condition, 'Madres menores de 30', group_name, year)

    # Comparar madres mayores de 30 con todos los grupos de comparación
    for group_name, condition in grupos_comparacion.items():
        calculate_odds_ratio_with_ci(group_data, madres_mayores_30, condition, 'Madres mayores de 30', group_name, year)

# Crear un DataFrame con los resultados y guardarlo en un archivo Excel
results_df = pd.DataFrame(results_list)
results_df.to_excel('resultados_odds_ratios_ci_por_ano.xlsx', index=False)
print("Resultados guardados en 'resultados_odds_ratios_ci_por_ano.xlsx'.")
