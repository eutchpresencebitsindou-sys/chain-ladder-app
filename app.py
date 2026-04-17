# ============================================================
# APP PROVISIONNEMENT ACTUARIEL
# Auteur : Eutch Présence BITSINDOU
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Provisionnement Actuariel", layout="wide")

st.title("📊 Provisionnement Actuariel")
st.subheader("Eutch Présence BITSINDOU")

# ============================================================
# 📥 IMPORT DES DONNÉES
# ============================================================

def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

uploaded_file = st.file_uploader("Importer un triangle (Excel ou CSV)")

manual_input = st.text_area("Ou coller un triangle (séparé par des espaces)")

triangle = None

if uploaded_file:
    triangle = load_data(uploaded_file)

elif manual_input:
    data = []
    for line in manual_input.strip().split("\n"):
        row = [float(x) if x.lower() != "nan" else np.nan for x in line.split()]
        data.append(row)
    triangle = pd.DataFrame(data)

if triangle is not None:
    st.write("Triangle chargé :")
    st.dataframe(triangle)

# ============================================================
# 🔄 INCREMENTAL -> CUMULÉ
# ============================================================

def to_cumulative(triangle):
    return triangle.cumsum(axis=1)

# ============================================================
# 🔧 FACTEURS CHAIN LADDER
# ============================================================

def compute_factors(triangle):
    n_cols = triangle.shape[1]
    factors = []

    for j in range(n_cols - 1):
        num = 0
        den = 0

        for i in range(triangle.shape[0]):
            if not pd.isna(triangle.iloc[i, j]) and not pd.isna(triangle.iloc[i, j + 1]):
                num += triangle.iloc[i, j + 1]
                den += triangle.iloc[i, j]

        factors.append(num / den if den != 0 else 1)

    return np.array(factors)

# ============================================================
# 🧠 CHAIN LADDER (CORRIGÉ)
# ============================================================

def complete_triangle_chain_ladder(triangle, factors):

    triangle = triangle.copy().astype(float)
    n_rows, n_cols = triangle.shape

    # sécurité critique
    if len(factors) != n_cols - 1:
        raise ValueError(f"Facteurs incorrects ({len(factors)} vs {n_cols-1})")

    for i in range(n_rows):
        for j in range(n_cols):

            if pd.isna(triangle.iloc[i, j]):

                prev = triangle.iloc[i, j - 1]

                if not pd.isna(prev):
                    triangle.iloc[i, j] = prev * factors[j - 1]

    latest = triangle.apply(lambda row: row.dropna().iloc[-1], axis=1)
    ultimate = triangle.iloc[:, -1]
    reserve = ultimate - latest

    return triangle, latest, ultimate, reserve

# ============================================================
# 📊 MACK
# ============================================================

def mack_std(triangle, factors):

    n = triangle.shape[1]
    sigma = []

    for j in range(n - 1):
        vals = []

        for i in range(triangle.shape[0] - j - 1):
            if not pd.isna(triangle.iloc[i, j]) and not pd.isna(triangle.iloc[i, j + 1]):
                ratio = triangle.iloc[i, j + 1] / triangle.iloc[i, j]
                vals.append((ratio - factors[j])**2 * triangle.iloc[i, j])

        sigma.append(np.sqrt(np.mean(vals)) if len(vals) > 0 else 0)

    return sigma

# ============================================================
# 🔁 BOOTSTRAP
# ============================================================

def bootstrap(triangle, factors, n_sim=500):

    sims = []

    for _ in range(n_sim):

        noise = np.random.normal(1, 0.05, len(factors))
        new_factors = factors * noise

        _, _, ultimate, _ = complete_triangle_chain_ladder(triangle, new_factors)
        sims.append(ultimate.sum())

    return np.array(sims)

# ============================================================
# 📈 LOSS RATIO
# ============================================================

def loss_ratio(primes, ratio):
    return primes * ratio

# ============================================================
# 📊 BORNHUETTER FERGUSON
# ============================================================

def bf_method(latest, primes, ratio, cdf):
    expected = primes * ratio
    return latest + expected * (1 - 1 / cdf)

# ============================================================
# 📊 BENKTANDER
# ============================================================

def benktander(cl, bf, cdf):
    return cl / cdf + bf * (1 - 1 / cdf)

# ============================================================
# 🚀 CALCUL PRINCIPAL
# ============================================================

if triangle is not None:

    mode = st.radio("Type de triangle", ["Cumulé", "Incrémental"])

    if mode == "Incrémental":
        triangle = to_cumulative(triangle)

    factors = compute_factors(triangle)

    st.write("Facteurs Chain-Ladder :", factors)

    completed, latest, ultimate, reserve = complete_triangle_chain_ladder(triangle, factors)

    st.subheader("Triangle complété")
    st.dataframe(completed)

    st.subheader("Résultats Chain-Ladder")
    st.write("Ultimate :", ultimate)
    st.write("Réserves :", reserve)
    st.write("Total réserve :", reserve.sum())

    # ========================================================
    # MACK
    # ========================================================
    sigma = mack_std(triangle, factors)
    st.subheader("Mack (volatilité)")
    st.write("Sigma :", sigma)

    # ========================================================
    # BOOTSTRAP
    # ========================================================
    sims = bootstrap(triangle, factors)

    st.subheader("Bootstrap")
    st.write("Moyenne :", np.mean(sims))
    st.write("Std :", np.std(sims))
    st.write("Quantile 95% :", np.quantile(sims, 0.95))

    # ========================================================
    # LOSS RATIO + BF + BENKTANDER
    # ========================================================

    st.subheader("Méthodes complémentaires")

    primes = st.number_input("Primes acquises", value=1000000.0)
    ratio = st.slider("Loss Ratio", 0.1, 1.0, 0.6)

    lr = loss_ratio(primes, ratio)
    st.write("Charge ultime (Loss Ratio) :", lr)

    cdf = np.prod(factors)

    bf = bf_method(latest.sum(), primes, ratio, cdf)
    st.write("BF :", bf)

    benk = benktander(ultimate.sum(), bf, cdf)
    st.write("Benktander :", benk)

    # ========================================================
    # EXPORT EXCEL
    # ========================================================
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        completed.to_excel(writer, sheet_name="Triangle")
        pd.DataFrame({"Ultimate": ultimate, "Reserve": reserve}).to_excel(writer, sheet_name="Results")

    st.download_button("📥 Télécharger Excel", buffer.getvalue(), file_name="provisionnement.xlsx")
