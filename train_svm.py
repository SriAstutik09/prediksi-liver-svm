"""
Melatih model SVM untuk prediksi penyakit liver, lalu menyimpannya ke file pickle.
Jalankan sekali:  python train_svm.py
Hasil: svm_liver_pickle (model) dan svm_liver_metrik.json (hasil evaluasi)
"""
import json
import os
import pickle

import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "Data_Pasien_penyakit_liver.csv")
MODEL_FILE = os.path.join(BASE, "svm_liver_pickle")
METRIK_FILE = os.path.join(BASE, "svm_liver_metrik.json")

FITUR = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
         "Alanine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Proteins",
         "Albumin", "Albumin_and_Globulin_Ratio"]


def muat_data():
    df = pd.read_csv(CSV).rename(columns={
        "Alamine_Aminotransferase": "Alanine_Aminotransferase",
        "Total_Protiens": "Total_Proteins"})
    df = df.drop_duplicates().reset_index(drop=True)   # sama seperti langkah EDA
    return df[FITUR], df["Dataset"]


def latih_dan_simpan():
    X, y = muat_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)

    # StandardScaler wajib karena skala fitur sangat berbeda (AST bisa ribuan, Albumin ~3)
    # class_weight="balanced" karena data tidak seimbang (71% pasien liver)
    pipa = Pipeline([("scaler", StandardScaler()),
                     ("svm", SVC(kernel="rbf", class_weight="balanced"))])
    grid = GridSearchCV(
        pipa,
        {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.01, 0.1]},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    metrik = {
        "akurasi": accuracy_score(y_test, y_pred),
        "akurasi_seimbang": balanced_accuracy_score(y_test, y_pred),
        "param_terbaik": {k: str(v) for k, v in grid.best_params_.items()},
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[1, 2]).tolist(),
        "laporan": classification_report(y_test, y_pred, output_dict=True),
        "n_train": len(X_train), "n_test": len(X_test),
        "median_fitur": X.median().to_dict(),
        "min_fitur": X.min().to_dict(), "max_fitur": X.max().to_dict(),
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(METRIK_FILE, "w") as f:
        json.dump(metrik, f, indent=2)
    return model, metrik


if __name__ == "__main__":
    model, m = latih_dan_simpan()
    print("Parameter terbaik :", m["param_terbaik"])
    print(f"Akurasi           : {m['akurasi']:.3f}")
    print(f"Akurasi seimbang  : {m['akurasi_seimbang']:.3f}")
    print("Confusion matrix [baris=aktual; kolom=prediksi] urutan (1=Liver, 2=Bukan):")
    print(m["confusion_matrix"])
    print(f"\nModel disimpan ke : {MODEL_FILE}")
