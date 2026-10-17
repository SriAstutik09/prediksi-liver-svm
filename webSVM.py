import json
import pickle

import pandas as pd
import streamlit as st

from train_models import FILE_MODEL, FITUR, METRIK_FILE, latih_dan_simpan

st.set_page_config(page_title="Prediksi Penyakit Liver", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")

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
/* ---- tambahan: pemilih algoritma, kartu putusan, tabel pemenang, bar kepentingan fitur ---- */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#B9C8DC !important; font-size:.8rem !important; }
.vcard { background:#fff; border:1px solid #E3E8EF; border-radius:16px; padding:1.1rem 1.2rem; box-shadow:0 2px 10px rgba(15,23,42,.04); height:100%; }
.vcard-k { color:#64748B; font-size:.78rem; font-weight:700; letter-spacing:.02em; margin-bottom:.5rem; }
.vcard-w { color:#0B2545; font-size:1.35rem; font-weight:800; letter-spacing:-.01em; line-height:1.15; }
.vcard-v { color:#334155; font-size:.88rem; font-weight:600; margin:.35rem 0 .5rem; }
.vcard-n { font-size:.78rem; font-weight:700; padding:.2rem .6rem; border-radius:999px; display:inline-block; }
.n-real { background:#E3F5EC; color:#0B7A53; } .n-tidak { background:#F1F5F9; color:#475569; }
.badge { white-space:nowrap; }
table.lab td.win { background:#E6F7F5 !important; color:#0B6B49; font-weight:800; }
table.lab th.c, table.lab td.c { text-align:center; }
.kes { color:#334155; font-size:.92rem; line-height:1.65; }
.kes b { color:#0B2545; }
.imp { display:flex; align-items:center; gap:.7rem; margin:.38rem 0; font-size:.84rem; color:#0F172A; font-weight:600; }
.imp-n { width:190px; flex:none; }
.imp-b { flex:1; background:#EEF2F6; border-radius:999px; height:10px; overflow:hidden; }
.imp-b span { display:block; height:100%; background:linear-gradient(90deg,#0E9F8E,#2DD4BF); border-radius:999px; }
.imp-v { width:46px; text-align:right; color:#64748B; font-weight:700; flex:none; }
.prob { margin:.9rem 0 .2rem; }
.prob-l { color:#334155; font-size:.84rem; font-weight:600; margin-bottom:.35rem; }
.prob-b { background:#EEF2F6; border-radius:999px; height:10px; overflow:hidden; }
.prob-b span { display:block; height:100%; background:linear-gradient(90deg,#F59E0B,#EF4444); border-radius:999px; }
.sama { background:#EEF6FF; border:1px solid #CFE2FA; color:#1E4E8C; border-radius:12px; padding:.7rem .95rem; font-size:.83rem; line-height:1.5; margin-bottom:1rem; }
.beda { background:#FFF8E6; border:1px solid #F3DFA2; color:#6B4E0B; border-radius:12px; padding:.7rem .95rem; font-size:.83rem; line-height:1.5; margin-bottom:1rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# Beri tahu browser bahwa halaman berbahasa Indonesia (mencegah terjemahan otomatis Edge/Chrome)
def atur_bahasa_id():
    js = ("<script>const d=window.parent.document;"
          "d.documentElement.lang='id';d.documentElement.setAttribute('translate','no');"
          "const m=d.createElement('meta');m.name='google';m.content='notranslate';d.head.appendChild(m);</script>")
    try:
        if hasattr(st, "iframe"):                 # Streamlit baru
            st.iframe(js, height=1)
        else:                                     # Streamlit lama
            import streamlit.components.v1 as components
            components.html(js, height=0)
    except Exception:
        pass


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
# Membaca model. Jika file pickle belum ada / versi scikit-learn berbeda sehingga gagal
# dibaca, kedua model dilatih ulang otomatis dari CSV (butuh sekitar 1-3 menit).
# ==============================================================================
@st.cache_resource(show_spinner="Menyiapkan model...")
def muat_model():
    try:
        model = {}
        for nama, path in FILE_MODEL.items():
            with open(path, "rb") as r:
                model[nama] = pickle.load(r)
        with open(METRIK_FILE) as f:
            metrik = json.load(f)
        return model, metrik
    except Exception:
        return latih_dan_simpan()


models, metrik = muat_model()
M = metrik["models"]
NAMA = ["SVM", "Random Forest"]
med = metrik["median_fitur"]
n_data = metrik["n_total"]
SEL = metrik["selisih_rf_minus_svm"]

pct = lambda v: f"{v * 100:.1f}%"
# model yang paling banyak menjaring pasien liver (recall tertinggi) -> pilihan awal untuk skrining
rekom = max(NAMA, key=lambda k: M[k]["recall_liver"])

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
      '<div class="brand-name">LiverCheck</div><div class="brand-sub">SVM &amp; Random Forest</div></div></div>')
    H('<div class="side-label">Menu</div>')
    choice = st.radio("Navigasi", ["Halaman Utama", "Perbandingan Model", "Tentang Kami", "Kontak"],
                      label_visibility="collapsed")
    H('<div class="side-label" style="margin-top:1.2rem">Algoritma prediksi</div>')
    pilih = st.selectbox("Algoritma prediksi", NAMA, index=NAMA.index(rekom), label_visibility="collapsed",
                         help="Pilihan awal adalah model yang paling banyak menjaring pasien liver (recall tertinggi).")
    H('<div class="side-foot">Aplikasi pembelajaran machine learning. '
      'Hasil prediksi bukan pengganti diagnosis dokter.</div>')


def prediksi(nama, nilai):
    """Mengembalikan (kelas, skor_liver). kelas 1 = pasien liver, 2 = bukan. skor hanya untuk model berprobabilitas."""
    x = pd.DataFrame([nilai], columns=FITUR)
    m = models[nama]
    kelas = int(m.predict(x)[0])
    skor = None
    if hasattr(m, "predict_proba"):
        skor = float(m.predict_proba(x)[0][list(m.classes_).index(1)])
    return kelas, skor


def fmt_acuan(lo, hi, f):
    return f"{f.format(lo)} – {f.format(hi)}"


def lencana_kelas(k):
    return ('<span class="badge b-hi">Pasien liver</span>' if k == 1
            else '<span class="badge b-ok">Bukan pasien</span>')


# ==============================================================================
# Halaman Utama
# ==============================================================================
if choice == 'Halaman Utama':
    hero("Support Vector Machine · Random Forest", "Prediksi Penyakit Liver",
         "Masukkan hasil pemeriksaan laboratorium pasien. Dua algoritma akan memberikan prediksi awal "
         "apakah pasien termasuk penderita penyakit liver atau bukan, lalu hasilnya bisa dibandingkan.",
         [(f"{n_data}", "data pasien"), ("10", "parameter input"), ("2", "algoritma")])

    kiri, kanan = st.columns([1.5, 1.1], gap="large")

    with kiri:
        try:
            form = st.form("form_pasien", border=False)
        except TypeError:                      # Streamlit versi lama
            form = st.form("form_pasien")
        with form:
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

            try:
                kirim = st.form_submit_button("Jalankan prediksi", width="stretch")
            except TypeError:                  # Streamlit versi lama
                kirim = st.form_submit_button("Jalankan prediksi", use_container_width=True)

        if kirim:
            st.session_state["nilai"] = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

    with kanan:
        nilai = st.session_state.get("nilai")
        if not nilai:
            H(f'<div class="panel kosong"><div class="kosong-ico">{IKON["doc"]}</div>'
              '<div class="kosong-t">Belum ada hasil</div>'
              '<div class="kosong-s">Isi data pasien di sebelah kiri, lalu tekan '
              '<b>Jalankan prediksi</b>. Nilai awal adalah median dari data pasien.</div></div>')
        else:
            hasil = {n: prediksi(n, nilai) for n in NAMA}
            kelas, skor = hasil[pilih]
            if kelas == 1:
                H(f'<div class="hasil hasil-liver"><div class="hasil-k">HASIL PREDIKSI · {pilih.upper()}</div>'
                  '<div class="hasil-t">Terindikasi pasien liver</div>'
                  '<div class="hasil-s">Kelas 1 · pola nilai laboratorium mirip dengan pasien liver pada data latih.</div></div>')
            else:
                H(f'<div class="hasil hasil-aman"><div class="hasil-k">HASIL PREDIKSI · {pilih.upper()}</div>'
                  '<div class="hasil-t">Tidak terindikasi pasien liver</div>'
                  '<div class="hasil-s">Kelas 2 · pola nilai laboratorium mirip dengan bukan pasien liver pada data latih.</div></div>')

            # --- perbandingan kedua model ---
            tandai = lambda n: ' <span style="color:#0E9F8E;font-weight:700">(dipakai)</span>' if n == pilih else ""
            baris_m = "".join(
                f'<tr><td>{n}{tandai(n)}</td>'
                f'<td>{lencana_kelas(hasil[n][0])}</td>'
                f'<td class="v">{pct(hasil[n][1]) if hasil[n][1] is not None else "–"}</td></tr>' for n in NAMA)
            sepakat = hasil["SVM"][0] == hasil["Random Forest"][0]
            ket = ('<div class="sama">Kedua algoritma <b>sepakat</b> pada hasil ini.</div>' if sepakat else
                   '<div class="beda">Kedua algoritma <b>memberi hasil berbeda</b>. Kasus seperti ini berada dekat batas '
                   'keputusan, jadi hasilnya perlu dibaca lebih hati-hati.</div>')
            H('<div class="panel"><div class="panel-title">Hasil kedua algoritma</div>'
              '<table class="lab"><tr><th>Algoritma</th><th>Prediksi</th><th>Skor liver</th></tr>'
              f'{baris_m}</table><div class="note">Skor liver hanya tersedia pada Random Forest dan merupakan '
              'skor model, bukan peluang klinis.</div></div>' + ket)

            if skor is not None:
                H(f'<div class="prob"><div class="prob-l">Skor pasien liver ({pilih}): <b>{pct(skor)}</b></div>'
                  f'<div class="prob-b"><span style="width:{skor * 100:.0f}%"></span></div></div>')

            baris, di_luar = "", 0
            for nama, i, sat, lo, hi, f in ACUAN:
                v = nilai[i]
                if v > hi:
                    badge, di_luar = '<span class="badge b-hi">Tinggi</span>', di_luar + 1
                elif v < lo:
                    badge, di_luar = '<span class="badge b-lo">Rendah</span>', di_luar + 1
                else:
                    badge = '<span class="badge b-ok">Normal</span>'
                baris += (f'<tr><td>{nama}</td><td class="v">{f.format(v)} <span style="color:#94A3B8;font-weight:500">{sat}</span></td>'
                          f'<td class="r">{fmt_acuan(lo, hi, f)}</td><td>{badge}</td></tr>')
            H('<div class="panel" style="margin-top:1rem"><div class="panel-title">Ringkasan nilai laboratorium</div>'
              f'<div style="color:#475569;font-size:.86rem;margin-bottom:.6rem"><b>{di_luar} dari {len(ACUAN)}</b> parameter berada di luar rentang acuan umum.</div>'
              '<table class="lab"><tr><th>Parameter</th><th>Nilai</th><th>Acuan</th><th></th></tr>'
              f'{baris}</table><div class="note">Rentang acuan bersifat umum untuk dewasa; nilai tiap laboratorium dapat berbeda.</div></div>')

            H('<div class="warn"><b>Perhatikan:</b> ini hasil model pembelajaran, bukan diagnosis medis. '
              f'Pada data uji, {pilih} mengenali sekitar {M[pilih]["recall_liver"] * 100:.0f}% pasien liver, sehingga hasil '
              '"tidak terindikasi" tetap perlu dikonfirmasi oleh dokter.</div>')

# ==============================================================================
# Perbandingan Model
# ==============================================================================
elif choice == 'Perbandingan Model':
    hero("Evaluasi", "Perbandingan SVM vs Random Forest",
         "Kedua algoritma dilatih dan diuji pada data yang sama, dengan pembagian data dan ukuran penilaian yang sama.")
    st.write("")

    def unggul(kunci):
        a, b = M["SVM"][kunci], M["Random Forest"][kunci]
        return None if abs(a - b) < 0.0005 else ("SVM" if a > b else "Random Forest")

    def kartu_putusan(judul, kunci, sel_kunci):
        w = unggul(kunci)
        s = SEL[sel_kunci]
        nyata = s["signifikan"]
        sel_txt = f"selisih {abs(s['nilai']) * 100:.1f} poin"
        badge = ('<span class="vcard-n n-real">Selisih nyata secara statistik</span>' if nyata
                 else '<span class="vcard-n n-tidak">Selisih tidak nyata</span>')
        H(f'<div class="vcard"><div class="vcard-k">{judul}</div>'
          f'<div class="vcard-w">{w if (w and nyata) else (w + " (tipis)" if w else "Seimbang")}</div>'
          f'<div class="vcard-v">SVM {pct(M["SVM"][kunci])} · RF {pct(M["Random Forest"][kunci])} · {sel_txt}</div>{badge}</div>')

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        kartu_putusan("AKURASI SEIMBANG (UKURAN UTAMA)", "akurasi_seimbang", "akurasi_seimbang")
    with c2:
        kartu_putusan("MENJARING PASIEN LIVER (RECALL)", "recall_liver", "recall_liver")
    with c3:
        kartu_putusan("MENGENALI BUKAN PASIEN (RECALL)", "recall_bukan", "recall_bukan")
    st.write("")

    # ---- kesimpulan otomatis ----
    sb = SEL["akurasi_seimbang"]
    rl, rb = unggul("recall_liver"), unggul("recall_bukan")
    if not sb["signifikan"]:
        k1 = (f"Secara keseluruhan kedua algoritma <b>setara</b>: selisih akurasi seimbang hanya {abs(sb['nilai']) * 100:.1f} poin "
              f"dan tidak nyata secara statistik (ROC-AUC SVM {M['SVM']['auc']:.3f} vs Random Forest {M['Random Forest']['auc']:.3f}).")
    else:
        k1 = f"<b>{metrik['best']}</b> unggul secara keseluruhan dengan selisih akurasi seimbang {abs(sb['nilai']) * 100:.1f} poin yang nyata secara statistik."
    k2 = (f"Perbedaannya ada pada <b>jenis kesalahan</b>: {rl} lebih banyak menjaring pasien liver "
          f"(recall {pct(M[rl]['recall_liver'])} vs {pct(M['SVM' if rl == 'Random Forest' else 'Random Forest']['recall_liver'])}), "
          f"sedangkan {rb} lebih tepat mengenali bukan pasien liver "
          f"(recall {pct(M[rb]['recall_bukan'])} vs {pct(M['SVM' if rb == 'Random Forest' else 'Random Forest']['recall_bukan'])}).") if rl and rb else ""
    k3 = (f"Untuk keperluan <b>skrining awal</b>, melewatkan pasien liver lebih merugikan daripada salah menandai orang sehat, "
          f"sehingga <b>{rekom}</b> dijadikan pilihan awal di halaman utama. Untuk keperluan yang butuh prediksi “pasien liver” "
          f"yang lebih bisa dipercaya (precision), pilih SVM.")
    H(f'<div class="panel"><div class="panel-title">Kesimpulan</div><div class="kes">{k1}<br><br>{k2}<br><br>{k3}</div></div>')

    # ---- tabel metrik ----
    BARIS = [("Akurasi", "akurasi", "p"), ("Akurasi seimbang", "akurasi_seimbang", "p"),
             ("Recall pasien liver", "recall_liver", "p"), ("Recall bukan pasien liver", "recall_bukan", "p"),
             ("Precision pasien liver", "precision_liver", "p"), ("Precision bukan pasien", "precision_bukan", "p"),
             ("F1-score (rata-rata)", "f1_macro", "p"), ("ROC-AUC", "auc", "d"),
             ("Akurasi seimbang validasi silang (data latih)", "cv_bacc", "p")]
    isi, menang = "", {"SVM": 0, "Random Forest": 0}
    for label, k, tipe in BARIS:
        w = unggul(k)
        if w:
            menang[w] += 1
        fm = (lambda v: pct(v)) if tipe == "p" else (lambda v: f"{v:.3f}")
        sel_s = ' class="win c"' if w == "SVM" else ' class="c"'
        sel_r = ' class="win c"' if w == "Random Forest" else ' class="c"'
        isi += f'<tr><td>{label}</td><td{sel_s}>{fm(M["SVM"][k])}</td><td{sel_r}>{fm(M["Random Forest"][k])}</td></tr>'
    ci_s, ci_r = M["SVM"]["ci_bacc"], M["Random Forest"]["ci_bacc"]
    H('<div class="panel"><div class="panel-title">Perbandingan ukuran kinerja (data uji)</div>'
      '<table class="lab"><tr><th>Ukuran</th><th class="c">SVM</th><th class="c">Random Forest</th></tr>'
      f'{isi}</table><div class="note">Sel berwarna hijau = nilai lebih tinggi pada ukuran tersebut. '
      f'SVM unggul pada {menang["SVM"]} ukuran, Random Forest pada {menang["Random Forest"]} ukuran. '
      f'Selang kepercayaan 95% akurasi seimbang: SVM {pct(ci_s[0])} – {pct(ci_s[1])}, Random Forest {pct(ci_r[0])} – {pct(ci_r[1])}.</div></div>')

    # ---- confusion matrix ----
    def cm_html(nama):
        cm = M[nama]["confusion_matrix"]
        return ('<div class="cm">'
                '<div class="h">Aktual \\ Prediksi</div><div class="h">Pasien liver</div><div class="h">Bukan pasien</div>'
                f'<div class="h">Pasien liver</div><div class="good">{cm[0][0]}</div><div class="bad">{cm[0][1]}</div>'
                f'<div class="h">Bukan pasien</div><div class="bad">{cm[1][0]}</div><div class="good">{cm[1][1]}</div></div>')
    a, b = st.columns(2, gap="large")
    with a:
        H(f'<div class="panel"><div class="panel-title">Confusion matrix · SVM</div>{cm_html("SVM")}</div>')
    with b:
        H(f'<div class="panel"><div class="panel-title">Confusion matrix · Random Forest</div>{cm_html("Random Forest")}</div>')

    # ---- selisih & fitur penting ----
    a, b = st.columns([1, 1], gap="large")
    with a:
        r = ""
        for label, k in [("Akurasi seimbang", "akurasi_seimbang"), ("Recall pasien liver", "recall_liver"),
                         ("Recall bukan pasien", "recall_bukan")]:
            s = SEL[k]
            bd = ('<span class="badge b-ok">Nyata</span>' if s["signifikan"] else '<span class="badge" style="background:#F1F5F9;color:#475569">Tidak nyata</span>')
            r += (f'<tr><td>{label}</td><td class="v">{s["nilai"] * 100:+.1f}</td>'
                  f'<td class="r">{s["ci"][0] * 100:+.1f} s.d. {s["ci"][1] * 100:+.1f}</td><td>{bd}</td></tr>')
        H('<div class="panel"><div class="panel-title">Uji selisih (Random Forest − SVM)</div>'
          '<table class="lab"><tr><th>Ukuran</th><th>Selisih (poin)</th><th>Selang 95%</th><th></th></tr>'
          f'{r}</table><div class="note">Dihitung dengan bootstrap 2000 kali pada data uji ({metrik["n_test"]} pasien). '
          'Jika selang tidak melewati 0, selisihnya nyata secara statistik; jika melewati 0, selisih bisa saja terjadi karena kebetulan.</div></div>')
        pt, pr_ = M["SVM"]["param_terbaik"], M["Random Forest"]["param_terbaik"]
        H('<div class="panel"><div class="panel-title">Parameter terbaik</div><table class="lab">'
          f'<tr><td>SVM</td><td>kernel RBF, C = {pt["C"]}, gamma = {pt["gamma"]}, class_weight = balanced</td></tr>'
          f'<tr><td>Random Forest</td><td>{pr_["n_estimators"]} pohon, max_depth = {pr_["max_depth"]}, '
          f'min_samples_leaf = {pr_["min_samples_leaf"]}, max_features = {pr_["max_features"]}, class_weight = balanced</td></tr></table></div>')
    with b:
        imp = M["Random Forest"]["importance"]
        mx = max(imp.values())
        NAMA_F = {"Age": "Usia", "Gender": "Jenis kelamin", "Total_Bilirubin": "Bilirubin total",
                  "Direct_Bilirubin": "Bilirubin direct", "Alkaline_Phosphotase": "ALP",
                  "Alanine_Aminotransferase": "ALT / SGPT", "Aspartate_Aminotransferase": "AST / SGOT",
                  "Total_Proteins": "Protein total", "Albumin": "Albumin", "Albumin_and_Globulin_Ratio": "Rasio A : G"}
        bars = "".join(f'<div class="imp"><div class="imp-n">{NAMA_F[k]}</div><div class="imp-b"><span style="width:{v / mx * 100:.0f}%"></span></div>'
                       f'<div class="imp-v">{v * 100:.1f}%</div></div>' for k, v in imp.items())
        H(f'<div class="panel"><div class="panel-title">Fitur terpenting menurut Random Forest</div>{bars}'
          '<div class="note">Menunjukkan seberapa besar tiap fitur dipakai Random Forest saat memisahkan kedua kelas. '
          'Ini bukan bukti sebab-akibat.</div></div>')

    H(f'<div class="warn"><b>Catatan:</b> data uji hanya {metrik["n_test"]} pasien, sehingga selisih kecil antar algoritma bisa terjadi karena kebetulan. '
      'Data juga tidak seimbang (±71% pasien liver), maka akurasi biasa bisa menyesatkan dan dipakai akurasi seimbang sebagai ukuran utama. '
      'Hasil ini hanya untuk pembelajaran, bukan untuk diagnosis.</div>')

# ==============================================================================
# Tentang Kami
# ==============================================================================
elif choice == 'Tentang Kami':
    hero("Tentang", "Tentang Kami", "Proyek pembelajaran machine learning dengan Streamlit.")
    H('<div class="panel"><div class="teks">Aplikasi ini dibuat untuk mempelajari dan membandingkan dua algoritma, '
      '<b>Support Vector Machine</b> dan <b>Random Forest</b>, pada data pasien penyakit liver. Data berisi hasil '
      'pemeriksaan laboratorium seperti bilirubin, enzim hati (ALP, ALT, AST), protein, dan albumin, dengan target '
      'pasien liver atau bukan.</div></div>')

# ==============================================================================
# Kontak
# ==============================================================================
elif choice == 'Kontak':
    hero("Hubungi kami", "Kontak", "Punya pertanyaan atau masukan? Silakan hubungi kami.")
    H(f'<div class="panel"><div class="kontak-row">{IKON["mail"]}<span>sri.astutik.student@unuja.ac.id</span></div></div>')
