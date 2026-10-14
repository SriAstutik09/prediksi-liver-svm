import json
import pickle

import pandas as pd
import streamlit as st

from train_svm import FITUR, METRIK_FILE, MODEL_FILE, latih_dan_simpan

st.set_page_config(page_title="Prediksi Penyakit Liver", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# TAMPILAN (CSS). Warna ditulis eksplisit supaya tampilan sama di mode terang/gelap.
# ==============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

.stApp, input, button, textarea, [data-baseweb="select"] {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif;
}
.stApp { background: #F4F7FA; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1240px; }

/* bersihkan elemen bawaan Streamlit */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
div[data-testid="stElementContainer"]:has(iframe) {
    position: absolute; height: 0; overflow: hidden; margin: 0;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2545 0%, #102F57 100%);
    border-right: none;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 0.5rem; }
.brand { display:flex; align-items:center; gap:12px; padding: 1.2rem 0.4rem 1.4rem; }
.brand-logo {
    width: 42px; height: 42px; border-radius: 12px; flex: none;
    background: linear-gradient(135deg, #14B8A6, #0E9F8E);
    display:flex; align-items:center; justify-content:center;
    box-shadow: 0 6px 16px rgba(20,184,166,.35);
}
.brand-name { color:#fff; font-weight:800; font-size:1.05rem; line-height:1.15; }
.brand-sub  { color:#93A9C4; font-size:.78rem; font-weight:500; }
.side-label { color:#7F95B2; font-size:.78rem; font-weight:600; letter-spacing:.02em; margin: .2rem 0 .5rem .4rem; }
.side-foot  { color:#7F95B2; font-size:.76rem; line-height:1.5; padding: 1.2rem .4rem 0; border-top:1px solid rgba(255,255,255,.1); margin-top:1.6rem; }

[data-testid="stSidebar"] [role="radiogroup"] { gap: .25rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: .65rem .9rem; border-radius: 10px; width: 100%; cursor: pointer;
    transition: background .15s;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child { display: none !important; }
[data-testid="stSidebar"] [role="radiogroup"] label p { color:#B9C8DC !important; font-weight:600; font-size:.95rem; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.07); }
[data-testid="stSidebar"] label[data-selected="true"] {
    background: rgba(20,184,166,.18); box-shadow: inset 3px 0 0 #2DD4BF;
}
[data-testid="stSidebar"] label[data-selected="true"] p { color:#fff !important; }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden; border-radius: 20px; padding: 2.1rem 2.3rem 1.9rem; margin-bottom: 1.4rem;
    background:
        radial-gradient(520px 260px at 92% -10%, rgba(45,212,191,.38), transparent 62%),
        repeating-linear-gradient(0deg, rgba(255,255,255,.035) 0 1px, transparent 1px 34px),
        linear-gradient(135deg, #0B2545 0%, #13315C 55%, #17476F 100%);
    box-shadow: 0 14px 34px rgba(11,37,69,.22);
}
.hero-tag { display:inline-block; font-size:.78rem; font-weight:700; color:#5EEAD4;
    background: rgba(94,234,212,.12); border:1px solid rgba(94,234,212,.3);
    padding:.28rem .7rem; border-radius:999px; margin-bottom:.9rem; }
.hero-title { color:#fff; font-size:2.15rem; font-weight:800; line-height:1.15; letter-spacing:-.02em; margin-bottom:.55rem; }
.hero-text  { color:#C3D2E6; font-size:1rem; line-height:1.6; max-width: 640px; margin-bottom:1.3rem; }
.hero-stats { display:flex; gap:2.4rem; flex-wrap:wrap; }
.hero-stats b { display:block; color:#fff; font-size:1.25rem; font-weight:800; }
.hero-stats span { color:#93A9C4; font-size:.8rem; font-weight:500; }

/* ---------- Kartu ---------- */
[class*="st-key-kartu_"] {
    background:#fff; border:1px solid #E3E8EF !important; border-radius:16px !important;
    padding: 1.15rem 1.3rem 1.1rem; box-shadow: 0 2px 10px rgba(15,23,42,.04);
    margin-bottom: .35rem;
}
.sec { display:flex; align-items:center; gap:.65rem; margin-bottom:.8rem; }
.sec-ico { width:30px; height:30px; border-radius:9px; background:#E6F7F5; display:flex; align-items:center; justify-content:center; flex:none; }
.sec-title { color:#0F172A; font-weight:700; font-size:1.02rem; line-height:1.2; }
.sec-sub { color:#64748B; font-size:.8rem; font-weight:500; }

/* ---------- Input ---------- */
[data-testid="stWidgetLabel"] p { color:#334155 !important; font-weight:600 !important; font-size:.86rem !important; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-testid="stNumberInputContainer"] {
    background:#fff !important; border-radius:10px !important;
}
[data-baseweb="input"] { border:1px solid #D5DCE6 !important; }
[data-baseweb="input"]:focus-within { border-color:#0E9F8E !important; box-shadow:0 0 0 3px rgba(14,159,142,.15) !important; }
[data-baseweb="input"] input, [data-baseweb="select"] div, [data-baseweb="select"] span {
    color:#0F172A !important; -webkit-text-fill-color:#0F172A !important; font-weight:600;
}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { background:#F1F5F9 !important; color:#475569 !important; }
[data-testid="stNumberInputStepDown"]:hover, [data-testid="stNumberInputStepUp"]:hover { background:#E2E8F0 !important; }
[data-baseweb="select"] > div { background:#fff !important; border:1px solid #D5DCE6 !important; border-radius:10px !important; }
[data-baseweb="select"] svg { fill:#475569 !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li { background:#fff !important; color:#0F172A !important; }
[data-baseweb="popover"] li:hover { background:#E6F7F5 !important; }

/* tombol prediksi */
[data-testid="stFormSubmitButton"] button {
    width:100%; height:3.1rem; border:none; border-radius:12px; font-weight:700; font-size:1rem;
    background: linear-gradient(90deg, #0E9F8E, #14B8A6); color:#fff !important;
    box-shadow: 0 8px 20px rgba(14,159,142,.28); transition: transform .12s, box-shadow .12s;
}
[data-testid="stFormSubmitButton"] button p { color:#fff !important; font-weight:700; }
[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(14,159,142,.38); }
[data-testid="stForm"] { border:none !important; padding:0 !important; }

/* ---------- Panel hasil ---------- */
.panel { background:#fff; border:1px solid #E3E8EF; border-radius:16px; padding:1.3rem 1.4rem; box-shadow:0 2px 10px rgba(15,23,42,.04); margin-bottom:1rem; }
.panel-title { color:#0F172A; font-weight:700; font-size:1.02rem; margin-bottom:.7rem; }
.kosong { text-align:center; padding:2.4rem 1.2rem; }
.kosong-ico { width:60px; height:60px; border-radius:18px; background:#E6F7F5; margin:0 auto 1rem; display:flex; align-items:center; justify-content:center; }
.kosong-t { color:#0F172A; font-weight:700; font-size:1.05rem; margin-bottom:.35rem; }
.kosong-s { color:#64748B; font-size:.9rem; line-height:1.55; }

.hasil { border-radius:16px; padding:1.4rem 1.5rem; margin-bottom:1rem; border:1px solid; }
.hasil-liver { background:#FDEEEE; border-color:#F5C2C2; }
.hasil-aman  { background:#E8F7F0; border-color:#B5E3CE; }
.hasil-k { font-size:.78rem; font-weight:700; letter-spacing:.02em; margin-bottom:.45rem; }
.hasil-liver .hasil-k { color:#B42318; } .hasil-aman .hasil-k { color:#0B7A53; }
.hasil-t { font-size:1.55rem; font-weight:800; line-height:1.2; letter-spacing:-.01em; margin-bottom:.4rem; }
.hasil-liver .hasil-t { color:#7A1B14; } .hasil-aman .hasil-t { color:#06432D; }
.hasil-s { font-size:.88rem; font-weight:500; line-height:1.5; }
.hasil-liver .hasil-s { color:#8F3B34; } .hasil-aman .hasil-s { color:#2F6B54; }

table.lab { width:100%; border-collapse:collapse; border:none !important; font-size:.84rem; }
table.lab th, table.lab td { border-left:none !important; border-right:none !important; border-top:none !important; background:transparent !important; }
table.lab td.v, table.lab td.r { white-space:nowrap; }
table.lab th { text-align:left; color:#64748B; font-weight:600; font-size:.76rem; padding:.35rem .4rem; border-bottom:1px solid #E3E8EF; }
table.lab td { padding:.55rem .4rem; border-bottom:1px solid #F0F3F7; color:#0F172A; font-weight:500; }
table.lab tr:last-child td { border-bottom:none; }
table.lab td.v { font-weight:700; }
table.lab td.r { color:#64748B; }
.badge { display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.74rem; font-weight:700; }
.b-ok { background:#E3F5EC; color:#0B7A53; } .b-hi { background:#FDE6E4; color:#B42318; } .b-lo { background:#FFF1D6; color:#92600A; }
.note { color:#64748B; font-size:.78rem; line-height:1.55; margin-top:.8rem; }
.warn { background:#FFF8E6; border:1px solid #F3DFA2; color:#6B4E0B; border-radius:12px; padding:.85rem 1rem; font-size:.84rem; line-height:1.55; }

/* ---------- Halaman lain ---------- */
.stat { background:#fff; border:1px solid #E3E8EF; border-radius:16px; padding:1.1rem 1.2rem; box-shadow:0 2px 10px rgba(15,23,42,.04); height:100%; }
.stat-v { font-size:1.8rem; font-weight:800; color:#0B2545; letter-spacing:-.02em; line-height:1.1; }
.stat-l { color:#64748B; font-size:.82rem; font-weight:600; margin-top:.3rem; }
.cm { display:grid; grid-template-columns: 1.1fr 1fr 1fr; gap:6px; font-size:.86rem; }
.cm div { padding:.75rem .6rem; border-radius:10px; text-align:center; }
.cm .h { background:#F1F5F9; color:#475569; font-weight:700; font-size:.78rem; }
.cm .good { background:#DDF3EA; color:#0B6B49; font-weight:800; font-size:1.35rem; }
.cm .bad  { background:#FCE4E2; color:#A3261C; font-weight:800; font-size:1.35rem; }
.step { display:flex; gap:.9rem; padding:.7rem 0; border-bottom:1px solid #F0F3F7; }
.step:last-child { border-bottom:none; }
.step-n { width:28px; height:28px; border-radius:50%; background:#0B2545; color:#fff; font-weight:700; font-size:.82rem; display:flex; align-items:center; justify-content:center; flex:none; }
.step-t { color:#0F172A; font-weight:700; font-size:.92rem; }
.step-d { color:#64748B; font-size:.84rem; line-height:1.5; }
.teks { color:#334155; font-size:.95rem; line-height:1.7; }
.kontak-row { display:flex; align-items:center; gap:.8rem; padding:.7rem 0; color:#0F172A; font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Beri tahu browser bahwa halaman berbahasa Indonesia (mencegah terjemahan otomatis Edge/Chrome)
def atur_bahasa_id():
    js = ("<script>const d=window.parent.document.documentElement;"
          "d.lang='id';d.setAttribute('translate','no');</script>")
    try:
        if hasattr(st, "iframe"):                 # Streamlit baru
            st.iframe(js, height=1)
        else:                                     # Streamlit lama
            import streamlit.components.v1 as components
            components.html(js, height=0)
    except Exception:
        pass                                      # tidak kritis; tampilan tetap jalan


atur_bahasa_id()


def H(html: str):
    """Tampilkan HTML (dibuat satu baris agar tidak dibaca sebagai blok kode markdown)."""
    st.markdown("".join(line.strip() for line in html.strip().splitlines()), unsafe_allow_html=True)


IKON = {
    "logo": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2.5-6 4 12 2.5-6H21"/></svg>',
    "user": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#0E9F8E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/></svg>',
    "flask": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#0E9F8E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6M10 3v6L4.5 19a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L14 9V3"/></svg>',
    "drop": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#0E9F8E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.5s6 6.2 6 11a6 6 0 0 1-12 0c0-4.8 6-11 6-11z"/></svg>',
    "doc": '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0E9F8E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/></svg>',
    "mail": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0E9F8E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/></svg>',
}


def judul_seksi(ikon, judul, sub=""):
    H(f'<div class="sec"><div class="sec-ico">{IKON[ikon]}</div><div><div class="sec-title">{judul}</div>'
      f'<div class="sec-sub">{sub}</div></div></div>')


def hero(tag, judul, teks, stats=None):
    s = ""
    if stats:
        s = '<div class="hero-stats">' + "".join(f"<div><b>{a}</b><span>{b}</span></div>" for a, b in stats) + "</div>"
    H(f'<div class="hero"><div class="hero-tag">{tag}</div><div class="hero-title">{judul}</div>'
      f'<div class="hero-text">{teks}</div>{s}</div>')


# ==============================================================================
# Membaca model (svm_liver_pickle). Jika file belum ada / versi scikit-learn
# berbeda sehingga gagal dibaca, model dilatih ulang otomatis dari CSV.
# ==============================================================================
@st.cache_resource(show_spinner="Menyiapkan model...")
def muat_model():
    try:
        with open(MODEL_FILE, "rb") as r:
            model = pickle.load(r)
        with open(METRIK_FILE) as f:
            metrik = json.load(f)
        return model, metrik
    except Exception:
        return latih_dan_simpan()


model, metrik = muat_model()
med = metrik["median_fitur"]
n_data = metrik["n_train"] + metrik["n_test"]
lap = metrik["laporan"]

# Rentang acuan umum (dewasa). Tiap laboratorium bisa berbeda.
ACUAN = [  # (nama, indeks input, satuan, batas bawah, batas atas, format)
    ("Bilirubin total", 2, "mg/dL", 0.1, 1.2, "{:.1f}"),
    ("Bilirubin direct", 3, "mg/dL", 0.0, 0.3, "{:.1f}"),
    ("ALP", 4, "IU/L", 44, 147, "{:.0f}"),
    ("ALT / SGPT", 5, "IU/L", 7, 56, "{:.0f}"),
    ("AST / SGOT", 6, "IU/L", 10, 40, "{:.0f}"),
    ("Protein total", 7, "g/dL", 6.0, 8.3, "{:.1f}"),
    ("Albumin", 8, "g/dL", 3.5, 5.0, "{:.1f}"),
    ("Rasio A : G", 9, "", 1.1, 2.5, "{:.2f}"),
]

# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    H(f'<div class="brand"><div class="brand-logo">{IKON["logo"]}</div><div>'
      '<div class="brand-name">LiverCheck</div><div class="brand-sub">Prediksi berbasis SVM</div></div></div>')
    H('<div class="side-label">Menu</div>')
    choice = st.radio("Navigasi", ["Halaman Utama", "Performa Model", "Tentang Kami", "Kontak"],
                      label_visibility="collapsed")
    H('<div class="side-foot">Aplikasi pembelajaran machine learning. '
      'Hasil prediksi bukan pengganti diagnosis dokter.</div>')


def prediction1(x):
    """x = list 10 fitur sesuai urutan FITUR. Mengembalikan 1 (Pasien Liver) atau 2 (Bukan)."""
    return int(model.predict(pd.DataFrame([x], columns=FITUR))[0])


def fmt_acuan(lo, hi, f):
    return f"{f.format(lo)} – {f.format(hi)}"


# ==============================================================================
# Halaman Utama
# ==============================================================================
if choice == 'Halaman Utama':
    hero("Support Vector Machine · Data Pasien Liver", "Prediksi Penyakit Liver",
         "Masukkan hasil pemeriksaan laboratorium pasien, lalu model SVM akan memberikan prediksi awal "
         "apakah pasien termasuk penderita penyakit liver atau bukan.",
         [(f"{n_data}", "data pasien"), ("10", "parameter input"), ("SVM RBF", "algoritma")])

    kiri, kanan = st.columns([1.5, 1.1], gap="large")

    with kiri:
        with st.form("form_pasien", border=False):
            with st.container(border=True, key="kartu_a"):
                judul_seksi("user", "Data pasien", "Informasi dasar")
                c1, c2 = st.columns(2)
                x1 = c1.number_input("Usia (tahun)", 1, 120, int(med["Age"]), step=1)
                jk = c2.selectbox("Jenis kelamin", ["Laki-laki", "Perempuan"])
                x2 = 1 if jk == "Laki-laki" else 0

            with st.container(border=True, key="kartu_b"):
                judul_seksi("flask", "Bilirubin & enzim hati", "Hasil tes fungsi hati")
                c1, c2, c3 = st.columns(3)
                x3 = c1.number_input("Bil. total (mg/dL)", 0.0, 100.0, float(med["Total_Bilirubin"]),
                                     step=0.1, format="%.1f", help="Rentang acuan umum: 0,1 – 1,2 mg/dL")
                x4 = c2.number_input("Bil. direct (mg/dL)", 0.0, 50.0, float(med["Direct_Bilirubin"]),
                                     step=0.1, format="%.1f", help="Rentang acuan umum: 0,0 – 0,3 mg/dL")
                x5 = c3.number_input("ALP (IU/L)", 0, 5000, int(med["Alkaline_Phosphotase"]), step=1,
                                     help="Alkaline phosphatase. Rentang acuan umum: 44 – 147 IU/L")
                c1, c2, c3 = st.columns(3)
                x6 = c1.number_input("ALT / SGPT (IU/L)", 0, 5000, int(med["Alanine_Aminotransferase"]), step=1,
                                     help="Rentang acuan umum: 7 – 56 IU/L")
                x7 = c2.number_input("AST / SGOT (IU/L)", 0, 10000, int(med["Aspartate_Aminotransferase"]), step=1,
                                     help="Rentang acuan umum: 10 – 40 IU/L")

            with st.container(border=True, key="kartu_c"):
                judul_seksi("drop", "Protein darah", "Albumin dan globulin")
                c1, c2, c3 = st.columns(3)
                x8 = c1.number_input("Protein (g/dL)", 0.0, 15.0, float(med["Total_Proteins"]),
                                     step=0.1, format="%.1f", help="Rentang acuan umum: 6,0 – 8,3 g/dL")
                x9 = c2.number_input("Albumin (g/dL)", 0.0, 10.0, float(med["Albumin"]),
                                     step=0.1, format="%.1f", help="Rentang acuan umum: 3,5 – 5,0 g/dL")
                x10 = c3.number_input("Rasio A : G", 0.0, 5.0, float(med["Albumin_and_Globulin_Ratio"]),
                                      step=0.05, format="%.2f", help="Rentang acuan umum: 1,1 – 2,5")

            kirim = st.form_submit_button("Jalankan prediksi", width="stretch")

        if kirim:
            nilai = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
            st.session_state["hasil"] = {"kelas": prediction1(nilai), "nilai": nilai}

    with kanan:
        h = st.session_state.get("hasil")
        if not h:
            H(f'<div class="panel kosong"><div class="kosong-ico">{IKON["doc"]}</div>'
              '<div class="kosong-t">Belum ada hasil</div>'
              '<div class="kosong-s">Isi data pasien di sebelah kiri, lalu tekan '
              '<b>Jalankan prediksi</b>. Nilai awal adalah median dari data pasien.</div></div>')
        else:
            if h["kelas"] == 1:
                H('<div class="hasil hasil-liver"><div class="hasil-k">HASIL PREDIKSI MODEL</div>'
                  '<div class="hasil-t">Terindikasi pasien liver</div>'
                  '<div class="hasil-s">Kelas 1 · pola nilai laboratorium mirip dengan pasien liver pada data latih.</div></div>')
            else:
                H('<div class="hasil hasil-aman"><div class="hasil-k">HASIL PREDIKSI MODEL</div>'
                  '<div class="hasil-t">Tidak terindikasi pasien liver</div>'
                  '<div class="hasil-s">Kelas 2 · pola nilai laboratorium mirip dengan bukan pasien liver pada data latih.</div></div>')

            baris, di_luar = "", 0
            for nama, i, sat, lo, hi, f in ACUAN:
                v = h["nilai"][i]
                if v > hi:
                    badge, di_luar = '<span class="badge b-hi">Tinggi</span>', di_luar + 1
                elif v < lo:
                    badge, di_luar = '<span class="badge b-lo">Rendah</span>', di_luar + 1
                else:
                    badge = '<span class="badge b-ok">Normal</span>'
                baris += (f'<tr><td>{nama}</td><td class="v">{f.format(v)} <span style="color:#94A3B8;font-weight:500">{sat}</span></td>'
                          f'<td class="r">{fmt_acuan(lo, hi, f)}</td><td>{badge}</td></tr>')
            H('<div class="panel"><div class="panel-title">Ringkasan nilai laboratorium</div>'
              f'<div style="color:#475569;font-size:.86rem;margin-bottom:.6rem"><b>{di_luar} dari {len(ACUAN)}</b> parameter berada di luar rentang acuan umum.</div>'
              '<table class="lab"><tr><th>Parameter</th><th>Nilai</th><th>Acuan</th><th></th></tr>'
              f'{baris}</table><div class="note">Rentang acuan bersifat umum untuk dewasa; nilai tiap laboratorium dapat berbeda.</div></div>')

            rec = lap["1"]["recall"] * 100
            H('<div class="warn"><b>Perhatikan:</b> ini hasil model pembelajaran, bukan diagnosis medis. '
              f'Pada data uji, model hanya mengenali sekitar {rec:.0f}% pasien liver, sehingga hasil '
              '"tidak terindikasi" tetap perlu dikonfirmasi oleh dokter.</div>')

# ==============================================================================
# Performa Model
# ==============================================================================
elif choice == 'Performa Model':
    hero("Evaluasi", "Performa Model",
         "Hasil pengujian model SVM pada data uji yang tidak dipakai saat pelatihan.")
    st.write("")
    s = st.columns(4, gap="medium")
    kartu = [(f"{metrik['akurasi'] * 100:.1f}%", "Akurasi"),
             (f"{metrik['akurasi_seimbang'] * 100:.1f}%", "Akurasi seimbang"),
             (f"{lap['1']['recall'] * 100:.1f}%", "Recall pasien liver"),
             (f"{lap['2']['recall'] * 100:.1f}%", "Recall bukan pasien liver")]
    for kol, (v, l) in zip(s, kartu):
        with kol:
            H(f'<div class="stat"><div class="stat-v">{v}</div><div class="stat-l">{l}</div></div>')
    st.write("")

    a, b = st.columns([1, 1], gap="large")
    cm = metrik["confusion_matrix"]
    with a:
        H('<div class="panel"><div class="panel-title">Confusion matrix</div>'
          '<div class="cm">'
          '<div class="h">Aktual \\ Prediksi</div><div class="h">Pasien liver</div><div class="h">Bukan pasien</div>'
          f'<div class="h">Pasien liver</div><div class="good">{cm[0][0]}</div><div class="bad">{cm[0][1]}</div>'
          f'<div class="h">Bukan pasien</div><div class="bad">{cm[1][0]}</div><div class="good">{cm[1][1]}</div>'
          '</div><div class="note">Hijau = prediksi benar, merah = prediksi salah.</div></div>')
        r1, r2 = lap["1"], lap["2"]
        H('<div class="panel"><div class="panel-title">Laporan klasifikasi</div><table class="lab">'
          '<tr><th>Kelas</th><th>Precision</th><th>Recall</th><th>F1</th><th>Jumlah</th></tr>'
          f'<tr><td>1 · Pasien liver</td><td>{r1["precision"]:.2f}</td><td>{r1["recall"]:.2f}</td><td>{r1["f1-score"]:.2f}</td><td>{r1["support"]:.0f}</td></tr>'
          f'<tr><td>2 · Bukan pasien</td><td>{r2["precision"]:.2f}</td><td>{r2["recall"]:.2f}</td><td>{r2["f1-score"]:.2f}</td><td>{r2["support"]:.0f}</td></tr>'
          '</table></div>')
    with b:
        pt = metrik["param_terbaik"]
        langkah = [
            ("Pembersihan data", f"13 baris duplikat dihapus sehingga tersisa {n_data} pasien."),
            ("Pembagian data", f"{metrik['n_train']} data latih dan {metrik['n_test']} data uji (70 : 30, stratified)."),
            ("Penskalaan fitur", "StandardScaler menyamakan skala fitur (AST bisa ribuan, albumin hanya sekitar 3)."),
            ("Model SVM", f"Kernel RBF, C = {pt['svm__C']}, gamma = {pt['svm__gamma']}, class_weight = balanced."),
        ]
        isi = "".join(f'<div class="step"><div class="step-n">{i}</div><div><div class="step-t">{t}</div>'
                      f'<div class="step-d">{d}</div></div></div>' for i, (t, d) in enumerate(langkah, 1))
        H(f'<div class="panel"><div class="panel-title">Alur pembuatan model</div>{isi}</div>')
        H('<div class="warn"><b>Catatan:</b> data tidak seimbang (±71% pasien liver), sehingga SVM biasa '
          'cenderung menebak semua orang sebagai pasien liver. Pembobotan kelas dipakai untuk mengatasinya. '
          'Model masih sering melewatkan pasien liver, jadi hasilnya hanya untuk pembelajaran.</div>')

# ==============================================================================
# Tentang Kami
# ==============================================================================
elif choice == 'Tentang Kami':
    hero("Tentang", "Tentang Kami", "Proyek pembelajaran machine learning dengan Streamlit.")
    H('<div class="panel"><div class="teks">Aplikasi ini dibuat untuk mempelajari penerapan algoritma '
      '<b>Support Vector Machine</b> pada data pasien penyakit liver. Data berisi hasil pemeriksaan '
      'laboratorium seperti bilirubin, enzim hati (ALP, ALT, AST), protein, dan albumin, dengan target '
      'pasien liver atau bukan.</div></div>')

# ==============================================================================
# Kontak
# ==============================================================================
elif choice == 'Kontak':
    hero("Hubungi kami", "Kontak", "Punya pertanyaan atau masukan? Silakan hubungi kami.")
    H(f'<div class="panel"><div class="kontak-row">{IKON["mail"]}<span>sri.astutik.student@unuja.ac.id</span></div></div>')
