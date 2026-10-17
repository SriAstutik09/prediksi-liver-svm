"""
Melatih dan membandingkan dua algoritma untuk prediksi penyakit liver:
  1. SVM (Support Vector Machine, kernel RBF)
  2. Random Forest

Keduanya memakai data yang SAMA, pembagian data yang SAMA, dan ukuran penilaian yang SAMA,
sehingga perbandingannya adil.

Jalankan:  python train_models.py
Hasil   :  svm_liver_pickle, rf_liver_pickle, metrik_perbandingan.json
"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "Data_Pasien_penyakit_liver.csv")
FILE_MODEL = {"SVM": os.path.join(BASE, "svm_liver_pickle"),
              "Random Forest": os.path.join(BASE, "rf_liver_pickle")}
METRIK_FILE = os.path.join(BASE, "metrik_perbandingan.json")

FITUR = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
         "Alanine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Proteins",
         "Albumin", "Albumin_and_Globulin_Ratio"]


def muat_data():
    df = pd.read_csv(CSV).rename(columns={
        "Alamine_Aminotransferase": "Alanine_Aminotransferase",
        "Total_Protiens": "Total_Proteins"})
    df = df.drop_duplicates().reset_index(drop=True)      # sama seperti langkah EDA
    return df[FITUR], df["Dataset"]


def skor_kelas_liver(model, X):
    """Skor 'kemungkinan pasien liver' (kelas 1) untuk menghitung ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, list(model.classes_).index(1)]
    return -model.decision_function(X)       # SVC: nilai positif = kelas kedua (2)


def hitung_metrik(model, grid, X_test, y_test, waktu):
    y_pred = model.predict(X_test)
    lap = classification_report(y_test, y_pred, output_dict=True)
    return {
        "akurasi": accuracy_score(y_test, y_pred),
        "akurasi_seimbang": balanced_accuracy_score(y_test, y_pred),
        "recall_liver": lap["1"]["recall"], "recall_bukan": lap["2"]["recall"],
        "precision_liver": lap["1"]["precision"], "precision_bukan": lap["2"]["precision"],
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "auc": roc_auc_score((y_test == 1).astype(int), skor_kelas_liver(model, X_test)),
        "cv_bacc": grid.best_score_,
        "param_terbaik": {k.split("__")[-1]: str(v) for k, v in grid.best_params_.items()},
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[1, 2]).tolist(),
        "laporan": lap, "waktu_latih_detik": waktu,
    }


def bootstrap(y_test, pred_svm, pred_rf, n=2000, seed=0):
    """
    Bootstrap pada data uji (2000 kali). Menghasilkan:
      - selang kepercayaan 95% akurasi seimbang tiap model
      - selang 95% selisih (RF - SVM) untuk akurasi seimbang, recall pasien liver, recall bukan pasien
    Jika selang selisih TIDAK melewati 0, selisihnya nyata secara statistik.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_test); ps = np.asarray(pred_svm); pr = np.asarray(pred_rf)

    def recall(yy, pp, kelas):
        m = yy == kelas
        return (pp[m] == kelas).mean()

    bs, br = [], []
    d_bacc, d_rl, d_rb = [], [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yy, a, c = y[i], ps[i], pr[i]
        bs.append(balanced_accuracy_score(yy, a)); br.append(balanced_accuracy_score(yy, c))
        d_bacc.append(br[-1] - bs[-1])
        d_rl.append(recall(yy, c, 1) - recall(yy, a, 1))
        d_rb.append(recall(yy, c, 2) - recall(yy, a, 2))
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    ppred = {"SVM": ps, "Random Forest": pr}
    selisih = {}
    for nama, d, f in [("akurasi_seimbang", d_bacc, lambda p: balanced_accuracy_score(y, p)),
                       ("recall_liver", d_rl, lambda p: recall(y, p, 1)),
                       ("recall_bukan", d_rb, lambda p: recall(y, p, 2))]:
        ci = pct(d)
        selisih[nama] = {"nilai": float(f(pr) - f(ps)), "ci": ci,
                         "signifikan": not (ci[0] <= 0 <= ci[1])}
    return pct(bs), pct(br), selisih


def latih_dan_simpan():
    X, y = muat_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    kandidat = {
        "SVM": GridSearchCV(
            Pipeline([("scaler", StandardScaler()),
                      ("svm", SVC(kernel="rbf", class_weight="balanced"))]),
            {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.01, 0.1]},
            scoring="balanced_accuracy", cv=cv),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=0, n_jobs=-1),
            {"n_estimators": [200, 500], "max_depth": [None, 6, 12],
             "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.5]},
            scoring="balanced_accuracy", cv=cv),
    }

    model, metrik = {}, {}
    for nama, grid in kandidat.items():
        t = time.time()
        grid.fit(X_train, y_train)
        model[nama] = grid.best_estimator_
        metrik[nama] = hitung_metrik(grid.best_estimator_, grid, X_test, y_test, time.time() - t)
        with open(FILE_MODEL[nama], "wb") as f:
            pickle.dump(grid.best_estimator_, f)

    rf = model["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=FITUR).sort_values(ascending=False)
    metrik["Random Forest"]["importance"] = imp.to_dict()

    ci_svm, ci_rf, selisih = bootstrap(y_test, model["SVM"].predict(X_test), rf.predict(X_test))
    metrik["SVM"]["ci_bacc"], metrik["Random Forest"]["ci_bacc"] = ci_svm, ci_rf

    terbaik = max(metrik, key=lambda k: metrik[k]["akurasi_seimbang"])
    hasil = {
        "best": terbaik,
        "selisih_rf_minus_svm": selisih,
        "n_train": len(X_train), "n_test": len(X_test), "n_total": len(X),
        "median_fitur": X.median().to_dict(),
        "min_fitur": X.min().to_dict(), "max_fitur": X.max().to_dict(),
        "models": metrik,
    }
    with open(METRIK_FILE, "w") as f:
        json.dump(hasil, f, indent=2)
    return model, hasil


if __name__ == "__main__":
    print("Melatih SVM dan Random Forest (sekitar 1-3 menit)...")
    _, h = latih_dan_simpan()
    print(f"\n{'Ukuran':<24}{'SVM':>12}{'Random Forest':>16}")
    for k, label in [("akurasi", "Akurasi"), ("akurasi_seimbang", "Akurasi seimbang"),
                     ("recall_liver", "Recall pasien liver"), ("recall_bukan", "Recall bukan pasien"),
                     ("precision_liver", "Precision pasien liver"), ("f1_macro", "F1 macro"),
                     ("auc", "ROC-AUC"), ("cv_bacc", "Akurasi seimbang CV")]:
        print(f"{label:<24}{h['models']['SVM'][k]:>12.3f}{h['models']['Random Forest'][k]:>16.3f}")
    print("\nParameter terbaik SVM          :", h["models"]["SVM"]["param_terbaik"])
    print("Parameter terbaik Random Forest:", h["models"]["Random Forest"]["param_terbaik"])
    print(f"\nSelang kepercayaan 95% akurasi seimbang: SVM {h['models']['SVM']['ci_bacc'][0]:.3f}-{h['models']['SVM']['ci_bacc'][1]:.3f}"
          f" | RF {h['models']['Random Forest']['ci_bacc'][0]:.3f}-{h['models']['Random Forest']['ci_bacc'][1]:.3f}")
    print("\nSelisih Random Forest - SVM (bootstrap 2000x, selang 95%):")
    for k, label in [("akurasi_seimbang", "Akurasi seimbang"), ("recall_liver", "Recall pasien liver"),
                     ("recall_bukan", "Recall bukan pasien")]:
        d = h["selisih_rf_minus_svm"][k]
        print(f"  {label:<22}{d['nilai']:+.3f}  [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  ->",
              "NYATA" if d["signifikan"] else "tidak nyata")
    print(f"\n>>> Model terbaik (akurasi seimbang tertinggi): {h['best']}")
    print("Model disimpan: svm_liver_pickle, rf_liver_pickle, metrik_perbandingan.json")
