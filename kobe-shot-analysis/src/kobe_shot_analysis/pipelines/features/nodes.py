import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    print("🔍 Iniciando engenharia de atributos...")

    # Exemplo: Distância ao aro
    if {"loc_x", "loc_y"}.issubset(df.columns):
        df["distance_to_hoop"] = np.sqrt(df["loc_x"] ** 2 + df["loc_y"] ** 2)

    # Exemplo: Arremesso de 3 pontos
    if "shot_type" in df.columns:
        df["is_three_point"] = df["shot_type"].apply(lambda x: 1 if "3PT" in x else 0)

    # Encoding: Variáveis categóricas com poucas categorias
    cat_cols = [
        col for col in df.select_dtypes(include="object") if df[col].nunique() <= 10
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Preencher nulos com mediana
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df
