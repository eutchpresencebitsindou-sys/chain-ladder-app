# ============================================================
#  PROVISIONNEMENT ACTUARIEL – APP STREAMLIT (AMÉLIORÉ)
#  Méthodes : Chain-Ladder | Bornhuetter–Ferguson | Mack | Bootstrap (ODP) | London-Chain
#  Auteur : Eutch Présence BITSINDOU
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Provisionnement actuariel | Eutch Présence BITSINDOU",
    layout="wide"
)

st.title("📊 Provisionnement actuariel – Assurance Non-Vie")
st.markdown("**Application développée par _Eutch Présence BITSINDOU_** — Chain-Ladder • BF • Mack • Bootstrap • London-Chain")

# =========================
# UTILITAIRES
# =========================

def ajouter_ligne_total(df: pd.DataFrame, colonnes_a_sommer: list[str], libelle_col0: str = "TOTAL") -> pd.DataFrame:
    """Ajoute une ligne TOTAL en bas d’un DataFrame (somme numérique pour colonnes choisies)."""
    if df is None or df.empty:
        return df
    total = df[colonnes_a_sommer].sum(numeric_only=True)
    ligne_total = {df.columns[0]: libelle_col0}
    for c in colonnes_a_sommer:
        ligne_total[c] = float(total.get(c, np.nan))
    return pd.concat([df, pd.DataFrame([ligne_total])], ignore_index=True)

def dernier_cumule_par_ligne(tri_cum: pd.DataFrame) -> pd.Series:
    """Renvoie le dernier cumul observé (diagonale) pour chaque origine (NaN si aucun)."""
    dev = tri_cum.columns[1:]
    return tri_cum[dev].apply(lambda r: r.dropna().iloc[-1] if r.dropna().size > 0 else np.nan, axis=1)

def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Export multi-feuilles Excel en mémoire (openpyxl)."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None:
                continue
            try:
                df.to_excel(writer, index=False, sheet_name=name[:31])
            except Exception:
                df2 = df.copy().astype(str)
                df2.to_excel(writer, index=False, sheet_name=name[:31])
    bio.seek(0)
    return bio.getvalue()

# =========================
# IMPORT / PARSING / TRANSFORMATIONS (CORRIGÉ)
# =========================

def nettoyer_triangle_df(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un triangle importé/collé sans casser la structure."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Supprimer lignes/colonnes vides
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Nettoyage noms colonnes
    df.columns = [str(c).strip() if str(c).strip() != "" else f"col_{i}" for i, c in enumerate(df.columns)]

    # Première colonne = origine
    col0 = df.columns[0]
    df[col0] = df[col0].fillna("").astype(str).str.strip()

    # Supprimer lignes vides
    df = df[df[col0] != ""]

    # Conversion des colonnes numériques
    for c in df.columns[1:]:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            .str.replace("\u00A0", "", regex=False)  # espace invisible
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)


def lire_triangle_fichier(fichier) -> pd.DataFrame:
    """Lecture robuste CSV / Excel"""
    nom = fichier.name.lower()

    try:
        if nom.endswith(".csv"):
            df = pd.read_csv(
                fichier,
                sep=None,
                engine="python",
                dtype=str,
                encoding="utf-8"
            )
        else:
            df = pd.read_excel(fichier, dtype=str)
    except Exception:
        fichier.seek(0)
        df = pd.read_csv(
            fichier,
            sep=None,
            engine="python",
            dtype=str,
            encoding="latin1"
        )

    return nettoyer_triangle_df(df)


def lire_triangle_texte(texte: str, sep_hint: str = None) -> pd.DataFrame:
    """Lecture robuste du collage"""
    txt = texte.strip()

    if not txt:
        return pd.DataFrame()

    if sep_hint is None:
        if "\t" in txt:
            sep = "\t"
        elif ";" in txt:
            sep = ";"
        elif "," in txt:
            sep = ","
        else:
            sep = r"\s+"
    else:
        sep = sep_hint

    df = pd.read_csv(
        io.StringIO(txt),
        sep=sep,
        engine="python",
        dtype=str
    )

    return nettoyer_triangle_df(df)


def incremental_vers_cumule(triangle_inc: pd.DataFrame) -> pd.DataFrame:
    """Conversion incrémental → cumulé propre"""
    tri = triangle_inc.copy()
    dev_cols = list(tri.columns[1:])

    for i in range(len(tri)):
        cumul = 0.0
        stop = False

        for col in dev_cols:
            val = tri.loc[i, col]

            if stop or pd.isna(val):
                tri.loc[i, col] = np.nan
                stop = True
            else:
                cumul += float(val)
                tri.loc[i, col] = cumul

    return tri


def cumule_vers_incremental(triangle_cum: pd.DataFrame) -> pd.DataFrame:
    """Conversion cumulé → incrémental"""
    tri = triangle_cum.copy()
    dev = list(tri.columns[1:])
    inc = tri.copy()

    for i in range(len(tri)):
        prev = np.nan

        for j, col in enumerate(dev):
            val = tri.loc[i, col]

            if pd.isna(val):
                inc.loc[i, col] = np.nan
            else:
                if j == 0:
                    inc.loc[i, col] = val
                else:
                    inc.loc[i, col] = val - prev

                prev = val

    return inc
