# ============================================================
#  PROVISIONNEMENT ACTUARIEL – APP STREAMLIT (CORRIGÉE)
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Provisionnement actuariel", layout="wide")

st.title("📊 Provisionnement actuariel – Non Vie")
st.markdown("Chain-Ladder • BF • Mack • Bootstrap • London-Chain")

# =========================
# NETTOYAGE (CORRECTION PRINCIPALE)
# =========================

def nettoyer_triangle_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Supprimer lignes/colonnes vides
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Colonnes propres
    df.columns = [str(c).strip() for c in df.columns]

    # Première colonne (origine)
    col0 = df.columns[0]
    df[col0] = df[col0].fillna("").astype(str).str.strip()

    df = df[df[col0] != ""]

    # Nettoyage colonnes numériques
    for c in df.columns[1:]:
        df[c] = (
            df[c].astype(str)
            .str.replace(" ", "")
            .str.replace(",", ".")
            .replace(["", "nan", "None"], np.nan)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)

# =========================
# IMPORT
# =========================

def lire_triangle_fichier(fichier):
    nom = fichier.name.lower()

    try:
        if nom.endswith(".csv"):
            df = pd.read_csv(fichier, sep=None, engine="python", dtype=str)
        else:
            df = pd.read_excel(fichier, dtype=str)
    except:
        fichier.seek(0)
        df = pd.read_csv(fichier, sep=None, engine="python", dtype=str, encoding="latin1")

    return nettoyer_triangle_df(df)

def lire_triangle_texte(txt, sep=None):
    txt = txt.strip()
    if not txt:
        return pd.DataFrame()

    if sep is None:
        if ";" in txt:
            sep = ";"
        elif "," in txt:
            sep = ","
        elif "\t" in txt:
            sep = "\t"
        else:
            sep = r"\s+"

    df = pd.read_csv(io.StringIO(txt), sep=sep, engine="python", dtype=str)
    return nettoyer_triangle_df(df)

# =========================
# TRANSFORMATION
# =========================

def incremental_vers_cumule(df):
    tri = df.copy()
    dev = tri.columns[1:]

    for i in range(len(tri)):
        cumul = 0
        stop = False

        for col in dev:
            val = tri.loc[i, col]

            if stop or pd.isna(val):
                tri.loc[i, col] = np.nan
                stop = True
            else:
                cumul += float(val)
                tri.loc[i, col] = cumul

    return tri

# =========================
# CHAIN LADDER
# =========================

def compute_factors(tri):
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

    return np.array(factors)

def project_triangle(tri, factors):
    tri = tri.copy()
    dev = tri.columns[1:]

    for i in range(len(tri)):
        for j in range(len(dev)-1):
            a = dev[j]
            b = dev[j+1]

            if pd.notna(tri.loc[i, a]) and pd.isna(tri.loc[i, b]):
                tri.loc[i, b] = tri.loc[i, a] * factors[j]

    return tri

def compute_ultimate(tri, factors):
    dev = tri.columns[1:]
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

    return np.array(ult)

# =========================
# UI
# =========================

mode = st.sidebar.radio("Mode", ["Fichier", "Coller"])

df = None

if mode == "Fichier":
    f = st.sidebar.file_uploader("Importer", type=["csv", "xlsx"])
    if f:
        df = lire_triangle_fichier(f)
else:
    txt = st.sidebar.text_area("Coller triangle")
    if txt:
        df = lire_triangle_texte(txt)

if df is None or df.empty:
    st.stop()

# Sécurité
if df.shape[1] < 2:
    st.error("Triangle invalide")
    st.stop()

st.subheader("Triangle importé")
st.dataframe(df)

tri = incremental_vers_cumule(df)

st.subheader("Triangle cumulé")
st.dataframe(tri)

# =========================
# CALCUL
# =========================

factors = compute_factors(tri)

st.subheader("Facteurs")
st.write(factors)

tri_proj = project_triangle(tri, factors)

st.subheader("Triangle projeté")
st.dataframe(tri_proj)

ult = compute_ultimate(tri, factors)

df_res = pd.DataFrame({
    "Année": tri.iloc[:,0],
    "Ultime": ult
})

df_res["PSAP"] = df_res["Ultime"] - tri.iloc[:,1:].max(axis=1)

st.subheader("Résultats")
st.dataframe(df_res)

st.metric("IBNR total", f"{df_res['PSAP'].sum():,.2f}")
