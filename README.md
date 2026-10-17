# Prediksi Penyakit Liver: SVM vs Random Forest

Aplikasi web Streamlit untuk memprediksi pasien liver dari hasil pemeriksaan laboratorium, dengan dua algoritma (**Support Vector Machine** dan **Random Forest**) yang bisa dibandingkan langsung.

**Aplikasi online:** (https://prediksi-penyakit-liver.streamlit.app/)

> Proyek pembelajaran machine learning. Hasil prediksi **bukan** diagnosis medis.

## Struktur proyek

```
prediksi-liver-svm/
├── webSVM.py                        aplikasi web (file utama yang dijalankan Streamlit)
├── train_models.py                  melatih dan membandingkan SVM vs Random Forest (diimpor oleh webSVM.py)
├── Data_Pasien_penyakit_liver.csv   dataset
├── svm_liver_pickle                 model SVM terlatih
├── rf_liver_pickle                  model Random Forest terlatih
├── metrik_perbandingan.json         hasil evaluasi kedua model
├── requirements.txt                 daftar library
├── .streamlit/config.toml           tema aplikasi
└── notebook/
    ├── 01_Train_Model_SVM_vs_RandomForest.ipynb   pelatihan model, langkah demi langkah
    └── 02_Aplikasi_Streamlit_webSVM.ipynb         penjelasan kode aplikasi per bagian
```

File `.py` tetap dibutuhkan karena Streamlit hanya bisa menjalankan file `.py`. Folder `notebook/` adalah versi belajar dan dokumentasi.

## Menjalankan di komputer

```
python -m pip install -r requirements.txt
python -m streamlit run webSVM.py
```

Untuk membuka notebook: `python -m pip install ipykernel matplotlib seaborn`, lalu buka file `.ipynb` di VS Code atau Jupyter.
