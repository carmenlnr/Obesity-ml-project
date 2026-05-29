# Predicción de Nivel de Obesidad — Proyecto ML
 
### Descripción del proyecto
 
Este proyecto construye un modelo de Machine Learning para predecir el nivel de obesidad de una persona basándose en sus hábitos de vida y condición física. La variable objetivo tiene 7 categorías, desde Peso Insuficiente hasta Obesidad Tipo III, lo que convierte este problema en una **clasificación multiclase** — el modelo predice a qué categoría pertenece cada persona, no un número continuo.
 
El dataset proviene del [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544) y contiene datos de personas de México, Perú y Colombia.
 
---
 
### Objetivo
 
Predecir el nivel de obesidad de una persona a partir de sus características físicas y hábitos de vida, probando diferentes modelos de Machine Learning, comparando su rendimiento y aplicando Hyperparameter Tuning sobre los mejores.
 
---
 
### Dataset
 
- **Fuente:** UCI Machine Learning Repository (ID: 544)
- **Tamaño:** 2.111 filas × 17 columnas
- **Target:** `obesity_level` — 7 categorías: Insufficient Weight, Normal Weight, Overweight Level I & II, Obesity Type I, II & III
- **Features:** Edad, altura, peso, hábitos alimenticios, actividad física, uso de tecnología, transporte, etc.
- **Valores nulos:** Ninguno — el dataset está limpio y balanceado


---
 
## Proceso
 
### 1. Carga y exploración de datos
- Carga del dataset directamente desde UCI con `fetch_ucirepo`
- Exploración del shape, tipos de datos y distribución del target
- Comprobación de valores nulos — el dataset estaba completamente limpio
### 2. Limpieza y preparación de datos
- Sin valores nulos, no fue necesario eliminar ni imputar filas
- Separación de features (X) y target (y)
- División train/test: 80% entrenamiento / 20% evaluación con `random_state=0`
### 3. Modelo base — KNN
- Primer modelo entrenado solo con las 8 features numéricas
- Resultado base: **85.6% de aciertos** en datos nuevos
### 4. Encoding de variables categóricas
- One-Hot Encoding con `pd.get_dummies` y `drop_first=True` para eliminar columnas innecesarias
- Se pasó de 8 features numéricas a 23 features en total
### 5. Normalización
- Escalado con `StandardScaler` para que todas las features estén en la misma escala
- Aplicado siempre después del split — `fit_transform` en train, solo `transform` en test
### 6. Modelos probados
 
 ![Tabla comparativa de modelos](tabla_comparativa.png)
 
### 7. Métricas de evaluación
- **Accuracy** → porcentaje global de predicciones correctas
- **F1-score** → combina precisión y recall por clase — útil para ver si el modelo funciona bien en todas las categorías
- **Overfitting** → diferencia entre train y test accuracy — si es muy grande el modelo ha memorizado los datos en lugar de aprender patrones reales
- **CV score** → resultado medio evaluando el modelo 5 veces con divisiones distintas — confirma que el resultado es consistente
### 8. Modelos principales y por qué
Se eligieron **Gradient Boosting** y **Logistic Regression** como modelos principales porque representan enfoques distintos y obtuvieron los mejores resultados:
- **Gradient Boosting** → mejor accuracy en datos nuevos (96.9%)
- **Logistic Regression** → mejor equilibrio entre aciertos y estabilidad (95.0%, solo 1.2pts de overfitting)
Se aplicó **Grid Search** y **Random Search** sobre ambos para optimizar sus hiperparámetros:
- Mejores parámetros GB: `learning_rate=0.2`, `max_depth=3`, `n_estimators=100`
- Mejores parámetros LR: `C=10`, `max_iter=500`


---
 
### Resultados/Insights
 
- **Mejor modelo:** Gradient Boosting con tuning — **96.9% de aciertos** en datos nuevos
- **Modelo más estable:** Logistic Regression con tuning — 95.0% con solo 1.2pts de overfitting
- **Features más importantes:** Weight, Height y Gender explican el 93% de las decisiones del modelo lo que tiene sentido ya que son los principales indicadores de obesidad
- La clase más difícil de predecir es **Normal_Weight**, el modelo la confunde con categorías cercanas como Insufficient_Weight y Overweight, lo cual tiene sentido porque son niveles muy parecidos entre sí.
- El dataset estaba balanceado y no fue necesario aplicar técnicas de balanceo
- **AdaBoost** no funcionó bien para este problema de 7 categorías (52% de aciertos)

### Conclusiones 
 - El tuning mejoró significativamente Logistic Regression (de 90.1% a 95.0%) pero apenas mejoró Gradient Boosting (de 96.7% a 96.9%) lo que sugiere que Gradient Boosting ya estaba cerca de su rendimiento óptimo con los parámetros por defecto
- Grid Search y Random Search encontraron los mismos parámetros óptimos — confirma que los hiperparámetros encontrados son sólidos
- No siempre el modelo más complejo es el mejor Logistic Regression con tuning (95%) supera a modelos ensemble como Random Forest (94.1%) siendo mucho más simple e interpretable

---
## Cómo replicar el proyecto
 
1. Clona el repositorio
2. Abre y ejecuta `notebooks/obesity_analysis.ipynb`

---
 
## Próximos pasos
 
- Aplicar `pd.cut()` para categorizar la edad en rangos y ver si mejora el modelo
- Probar grids más amplios en el hyperparameter tuning
- Construir una app con Streamlit para predecir el nivel de obesidad en tiempo real
- Explorar más en profundidad la importancia de las features para reducir la dimensionalidad
- Probar el modelo con datos de población europea, especialmente países mediterráneos como España, Portugal o Francia
---
 
## 👩‍💻 Autora
 
Carmen — Data Analytics Bootcamp, Mayo 2026

### Link presentación
[Link google slides ML Obesidad](https://docs.google.com/presentation/d/1UkmoT2i79t2TbimzNvopK_f_s1O_-PWzcF8OdaXN22Q/edit?usp=sharing)