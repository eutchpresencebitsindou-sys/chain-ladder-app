# ============================================================
#  PROVISIONNEMENT ACTUARIEL – APP COMPLETE
#  Méthodes : Chain-Ladder Standard | Mack | Bootstrap ODP
#  Auteur : Eutch Présence BITSINDOU
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Provisionnement actuariel",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Provisionnement Actuariel – Assurance Non-Vie")
st.markdown("### 👤 Eutch Présence BITSINDOU")

# ============================================================
# NETTOYAGE
# ============================================================

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

# ============================================================
# IMPORT
# ============================================================

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

# ============================================================
# TRANSFORMATION
# ============================================================

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

# ============================================================
# CHAIN LADDER STANDARD
# ============================================================

def chain_ladder_standard(tri):
    dev = tri.columns[1:]
    factors = []

    # FACTEURS STANDARD
    for j in range(len(dev)-1):
        a = dev[j]
        b = dev[j+1]

        valid = tri[[a,b]].dropna()
        if valid.empty:
            factors.append(np.nan)
        else:
            f = valid[b].sum() / valid[a].sum()
            factors.append(f)

    factors = np.array(factors)

    # PROJECTION
    tri_proj = tri.copy()
    for i in range(len(tri)):
        for j in range(len(dev)-1):
            if pd.notna(tri_proj.iloc[i,j+1]) and pd.isna(tri_proj.iloc[i,j+2]):
                tri_proj.iloc[i,j+2] = tri_proj.iloc[i,j+1] * factors[j]

    # ULTIMES
    ult = []
    for i in range(len(tri)):
        row = tri.iloc[i,1:]
        last = row.last_valid_index()

        if last is None:
            ult.append(np.nan)
        else:
            pos = list(dev).index(last)
            val = row[last]

            if pos < len(factors):
                prod = np.prod(factors[pos:])
            else:
                prod = 1

            ult.append(val * prod)

    return tri_proj, factors, np.array(ult)

# ============================================================
# MACK
# ============================================================

def mack(tri, factors):
    dev = tri.columns[1:]
    sigma2 = []

    for j in range(len(factors)):
        a = dev[j]
        b = dev[j+1]

        tmp = tri[[a,b]].dropna()
        tmp = tmp[tmp[a] > 0]

        if len(tmp) <= 1:
            sigma2.append(np.nan)
            continue

        ratios = tmp[b] / tmp[a]
        fj = factors[j]

        num = np.sum(tmp[a] * (ratios - fj)**2)
        sigma2.append(num / (len(tmp)-1))

    sigma2 = np.array(sigma2)

    # SE par origine
    se = []
    tri_proj, _, ult = chain_ladder_standard(tri)

    for i in range(len(tri)):
        var = 0
        row = tri.iloc[i,1:]
        last = row.last_valid_index()

        if last is None:
            se.append(np.nan)
            continue

        pos = list(dev).index(last)

        for k in range(pos, len(factors)):
            c = tri_proj.iloc[i,k+1]
            if pd.isna(c) or pd.isna(sigma2[k]):
                continue
            var += (c**2) * sigma2[k]

        se.append(np.sqrt(var))

    return np.array(se)

# ============================================================
# BOOTSTRAP ODP
# ============================================================

def bootstrap(tri, n_sim=1000):
    dev = tri.columns[1:]

    tri_proj, factors, _ = chain_ladder_standard(tri)

    inc_obs = tri.diff(axis=1)
    inc_obs.iloc[:,0] = tri.iloc[:,1]

    inc_exp = tri_proj.diff(axis=1)
    inc_exp.iloc[:,0] = tri_proj.iloc[:,1]

    resid = (inc_obs - inc_exp) / np.sqrt(inc_exp)
    resid = resid.stack().dropna().values

    ultimates = []

    for _ in range(n_sim):
        r = np.random.choice(resid, size=len(resid), replace=True)

        inc_sim = inc_exp.stack().values + r * np.sqrt(inc_exp.stack().values)
        inc_sim = np.maximum(inc_sim, 0)

        inc_sim = inc_sim.reshape(inc_exp.shape)
        tri_sim = np.cumsum(inc_sim, axis=1)

        ultimates.append(np.sum(tri_sim[:,-1]))

    return np.array(ultimates)

# ============================================================
# UI
# ============================================================

mode = st.sidebar.radio("Entrée", ["Fichier", "Coller"])
df = None

if mode == "Fichier":
    f = st.sidebar.file_uploader("Importer", type=["csv","xlsx"])
    if f:
        df = read_file(f)
else:
    txt = st.sidebar.text_area("Coller triangle")
    if txt:
        df = read_text(txt)

if df is None or df.empty:
    st.stop()

st.subheader("Triangle brut")
st.dataframe(df)

tri = incremental_to_cum(df)

st.subheader("Triangle cumulé")
st.dataframe(tri)

# ============================================================
# MÉTHODES
# ============================================================

method = st.sidebar.selectbox("Méthode", ["Chain-Ladder", "Mack", "Bootstrap"])

tri_proj, factors, ult = chain_ladder_standard(tri)
psap = ult - tri.iloc[:,1:].max(axis=1)

df_res = pd.DataFrame({
    "Année": tri.iloc[:,0],
    "Ultime": ult,
    "PSAP": psap
})

st.subheader("Résultats")
st.dataframe(df_res)

st.metric("IBNR Total", f"{psap.sum():,.2f}")

# ============================================================
# MACK
# ============================================================

if method == "Mack":
    se = mack(tri, factors)
    df_res["SE"] = se
    st.subheader("Mack (écart-type)")
    st.dataframe(df_res)

# ============================================================
# BOOTSTRAP
# ============================================================

if method == "Bootstrap":
    st.subheader("Bootstrap ODP")
    sims = bootstrap(tri, 1000)

    st.write("Moyenne :", np.mean(sims))
    st.write("Quantile 95% :", np.percentile(sims,95))

    fig, ax = plt.subplots()
    ax.hist(sims, bins=40)
    st.pyplot(fig)
