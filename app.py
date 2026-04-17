# ============================================================
#  PROVISIONNEMENT ACTUARIEL – APP PRO
#  Auteur : Eutch Présence BITSINDOU
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Provisionnement actuariel",
    layout="wide",
    page_icon="📊"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #00D4FF;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Provisionnement Actuariel")
st.markdown("### 👤 Eutch Présence BITSINDOU")

# =========================
# NETTOYAGE DATA
# =========================

def clean_df(df):
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    col0 = df.columns[0]
    df[col0] = df[col0].astype(str).str.strip()
    df = df[df[col0] != ""]

    for c in df.columns[1:]:
        df[c] = (
            df[c].astype(str)
            .str.replace(" ", "")
            .str.replace(",", ".")
            .replace(["", "nan"], np.nan)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)

# =========================
# IMPORT
# =========================

def read_file(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file, sep=None, engine="python", dtype=str)
    else:
        df = pd.read_excel(file, dtype=str)
    return clean_df(df)

def read_text(txt):
    if ";" in txt:
        sep = ";"
    elif "," in txt:
        sep = ","
    else:
        sep = r"\s+"

    df = pd.read_csv(io.StringIO(txt), sep=sep, engine="python", dtype=str)
    return clean_df(df)

# =========================
# TRANSFORMATIONS
# =========================

def incremental_to_cum(df):
    tri = df.copy()
    dev = tri.columns[1:]

    for i in range(len(tri)):
        cum = 0
        stop = False
        for c in dev:
            val = tri.loc[i, c]
            if stop or pd.isna(val):
                tri.loc[i, c] = np.nan
                stop = True
            else:
                cum += float(val)
                tri.loc[i, c] = cum
    return tri

# =========================
# CHAIN LADDER
# =========================

def chain_ladder(tri):
    dev = tri.columns[1:]
    factors = []

    for j in range(len(dev)-1):
        a = dev[j]
        b = dev[j+1]

        mask = tri[a].notna() & tri[b].notna() & (tri[a] > 0)
        tmp = tri.loc[mask, [a, b]]

        if len(tmp) == 0:
            factors.append(np.nan)
        else:
            factors.append(tmp[b].sum() / tmp[a].sum())

    factors = np.array(factors)

    # Projection
    tri_proj = tri.copy()
    for i in range(len(tri)):
        for j in range(len(dev)-1):
            a = dev[j]
            b = dev[j+1]
            if pd.notna(tri_proj.loc[i, a]) and pd.isna(tri_proj.loc[i, b]):
                tri_proj.loc[i, b] = tri_proj.loc[i, a] * factors[j]

    # Ultimes
    ult = []
    for i in range(len(tri)):
        row = tri.loc[i, dev]
        last = row.last_valid_index()
        if last is None:
            ult.append(np.nan)
        else:
            pos = list(dev).index(last)
            val = row[last]
            prod = np.prod(factors[pos:]) if pos < len(factors) else 1
            ult.append(val * prod)

    return tri_proj, factors, np.array(ult)

# =========================
# BF
# =========================

def bornhuetter(tri, ult_apriori, cdf):
    dev = tri.columns[1:]
    res = []

    for i in range(len(tri)):
        row = tri.loc[i, dev]
        last = row.last_valid_index()

        if last is None:
            res.append(np.nan)
        else:
            c_obs = row[last]
            cdf_j = cdf[list(dev).index(last)]
            psap = (1 - 1/cdf_j) * ult_apriori[i]
            res.append(c_obs + psap)

    return np.array(res)

# =========================
# SIDEBAR
# =========================

mode = st.sidebar.radio("Entrée", ["Fichier", "Coller"])

df = None

if mode == "Fichier":
    f = st.sidebar.file_uploader("Importer", type=["csv", "xlsx"])
    if f:
        df = read_file(f)
else:
    txt = st.sidebar.text_area("Coller triangle")
    if txt:
        df = read_text(txt)

if df is None or df.empty:
    st.stop()

# =========================
# DATA
# =========================

st.subheader("📥 Triangle brut")
st.dataframe(df)

tri = incremental_to_cum(df)

st.subheader("📊 Triangle cumulé")
st.dataframe(tri)

# =========================
# MÉTHODES
# =========================

method = st.sidebar.selectbox("Méthode", ["Chain-Ladder", "Bornhuetter-Ferguson"])

tri_proj, factors, ult = chain_ladder(tri)

psap = ult - tri.iloc[:,1:].max(axis=1)

# =========================
# RESULTATS
# =========================

st.subheader("📈 Résultats")

df_res = pd.DataFrame({
    "Année": tri.iloc[:,0],
    "Ultime": ult,
    "PSAP": psap
})

st.dataframe(df_res)

st.metric("💰 IBNR Total", f"{psap.sum():,.2f}")

# =========================
# GRAPHIQUES
# =========================

st.subheader("📉 Visualisation")

fig = px.line(df_res, x="Année", y="Ultime", markers=True)
st.plotly_chart(fig, use_container_width=True)

fig2 = px.bar(df_res, x="Année", y="PSAP")
st.plotly_chart(fig2, use_container_width=True)
