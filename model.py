# Script py model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def entrenar_knn(df, target_col, test_size=0.2, random_state=0):
    # Separar features y target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Entrenar KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)

    print("Train accuracy:", knn.score(X_train_scaled, y_train))
    print("Test accuracy:", knn.score(X_test_scaled, y_test))

    return knn, X_train_scaled, X_test_scaled, y_train, y_test

def entrenar_ensemble(modelo, nombre, X_train, X_test, y_train, y_test, resultados):
    # Training modelo
    modelo.fit(X_train, y_train)

    # Evaluar modelo
    pred = modelo.predict(X_test)
    print(f"--- {nombre} ---")
    print("Train accuracy:", modelo.score(X_train, y_train))
    print("Test accuracy:", modelo.score(X_test, y_test))
    print(classification_report(y_test, pred))

    # Guardar métricas
    resultados[nombre] = {
        "train": round(modelo.score(X_train, y_train) * 100, 1),
        "test": round(modelo.score(X_test, y_test) * 100, 1)
    }

    return modelo