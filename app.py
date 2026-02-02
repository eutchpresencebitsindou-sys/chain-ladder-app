# ============================================================
#  PROVISIONNEMENT ACTUARIEL COMPLET
#  Méthodes : Chain-Ladder | Bornhuetter–Ferguson | Mack | Bootstrap
#  Auteur : Eutch Présence BITSINDOU
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Provisionnement actuariel | Eutch Présence BITSINDOU",
    layout="wide"
)

st.title("📊 Provisionnement actuariel – Assurance Non-Vie")
st.markdown("""
**Application actuarielle développée par _Eutch Présence BITSINDOU_**  
Méthodes : **Chain-Ladder • Bornhuetter–Ferguson • Mack • Bootstrap**
""")

# ============================================================
# UTILITAIRES D'AFFICHAGE
# ============================================================

def ajouter_ligne_total(df: pd.DataFrame, colonnes_a_sommer: list[str], libelle_col0: str = "TOTAL") -> pd.DataFrame:
    """Ajoute une ligne TOTAL en bas d’un DataFrame."""
    if df is None or df.empty:
        return df
    total = df[colonnes_a_sommer].sum(numeric_only=True)
    ligne_total = {df.columns[0]: libelle_col0}
    for c in colonnes_a_sommer:
        ligne_total[c] = float(total.get(c, np.nan))
    return pd.concat([df, pd.DataFrame([ligne_total])], ignore_index=True)

def last_observed_per_row(tri_cum: pd.DataFrame) -> pd.Series:
    """Dernier cumul observé (diagonale) par ligne."""
    dev = tri_cum.columns[1:]
    return tri_cum[dev].apply(lambda r: r.dropna().iloc[-1] if r.dropna().shape[0] > 0 else np.nan, axis=1)

# ============================================================
# FONCTIONS D’IMPORT & PRÉPARATION
# ============================================================

def lire_triangle_fichier(fichier) -> pd.DataFrame:
    """Lecture d’un triangle depuis un fichier CSV ou Excel."""
    nom = fichier.name.lower()
    if nom.endswith(".csv"):
        df = pd.read_csv(fichier, sep=None, engine="python")
    elif nom.endswith(".xlsx") or nom.endswith(".xls"):
        df = pd.read_excel(fichier)
    else:
        raise ValueError("Format de fichier non supporté")

    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def lire_triangle_texte(texte: str) -> pd.DataFrame:
    """Lecture d’un triangle collé (copié depuis Excel)."""
    texte = texte.strip()
    if not texte:
        return pd.DataFrame()

    if "\t" in texte:
        sep = "\t"
    elif ";" in texte:
        sep = ";"
    elif "," in texte:
        sep = ","
    else:
        sep = r"\s+"

    df = pd.read_csv(io.StringIO(texte), sep=sep, engine="python")
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def incremental_vers_cumule(triangle_inc: pd.DataFrame) -> pd.DataFrame:
    """
    Conversion incrémental → cumulé (règle actuarielle stricte) :
    - on cumule jusqu’au premier NA
    - après un NA, tout reste NA (pas de cumul avec des vides)
    """
    tri = triangle_inc.copy()
    dev_cols = tri.columns[1:]

    for i in range(len(tri)):
        cumul = 0.0
        stop = False
        for col in dev_cols:
            val = tri.loc[i, col]
            if stop or pd.isna(val):
                tri.loc[i, col] = np.nan
                stop = True
            else:
                cumul += val
                tri.loc[i, col] = cumul
    return tri

# ============================================================
# CHAIN-LADDER CLASSIQUE (briques communes)
# ============================================================

def calcul_facteurs_chain_ladder(triangle_cum: pd.DataFrame) -> pd.Series:
    """Facteurs âge-à-âge Chain-Ladder (moyenne pondérée officielle)."""
    dev_cols = list(triangle_cum.columns[1:])
    facteurs = []

    for j in range(len(dev_cols) - 1):
        a, b = dev_cols[j], dev_cols[j + 1]
        mask = triangle_cum[a].notna() & triangle_cum[b].notna()
        num = triangle_cum.loc[mask, b].sum()
        den = triangle_cum.loc[mask, a].sum()
        facteurs.append(num / den if den and den > 0 else np.nan)

    return pd.Series(facteurs, index=dev_cols[:-1], name="Facteur CL")

def calcul_cdf(facteurs: pd.Series, dev_cols: list[str]) -> pd.Series:
    """CDF vers l'ultime : produit des facteurs restants (dernier = 1)."""
    cdf = []
    for j in range(len(dev_cols)):
        if j < len(facteurs):
            cdf.append(float(np.prod(facteurs.iloc[j:])))
        else:
            cdf.append(1.0)
    return pd.Series(cdf, index=dev_cols, name="CDF")

def projection_triangle_chain_ladder(triangle_cum: pd.DataFrame, facteurs: pd.Series) -> pd.DataFrame:
    """Complétion pédagogique du triangle cumulé (séquentielle)."""
    tri = triangle_cum.copy()
    dev_cols = list(tri.columns[1:])

    for i in range(len(tri)):
        for j in range(len(dev_cols) - 1):
            a, b = dev_cols[j], dev_cols[j + 1]
            if pd.notna(tri.loc[i, a]) and pd.isna(tri.loc[i, b]):
                tri.loc[i, b] = tri.loc[i, a] * facteurs.iloc[j]
    return tri

def calcul_ultimes_chain_ladder(triangle_cum: pd.DataFrame, facteurs: pd.Series) -> pd.Series:
    """
    Ultime Chain-Ladder (formule officielle) :
    Ultime_i = dernier cumul observé × produit des facteurs restants
    """
    dev_cols = list(triangle_cum.columns[1:])
    ult = []

    for i in range(len(triangle_cum)):
        ligne = triangle_cum.loc[i, dev_cols]
        last_col = ligne.last_valid_index()
        if last_col is None:
            ult.append(np.nan)
            continue
        pos = dev_cols.index(last_col)
        cumul = ligne[last_col]
        prod_rest = float(np.prod(facteurs.iloc[pos:])) if pos < len(facteurs) else 1.0
        ult.append(cumul * prod_rest)

    return pd.Series(ult, name="Ultime CL")

def calcul_psap(triangle_cum: pd.DataFrame, ultimes: pd.Series, label_ultime: str = "Ultime") -> pd.DataFrame:
    """PSAP = Ultime − Dernier règlement observé (diagonale)."""
    dernier = last_observed_per_row(triangle_cum)
    return pd.DataFrame({
        "Année de survenance": triangle_cum.iloc[:, 0].astype(str),
        "Dernier règlement observé": dernier.values,
        label_ultime: ultimes.values,
        "PSAP": (ultimes.values - dernier.values)
    })

# ============================================================
# BORNHUETTER–FERGUSON
# ============================================================

def calcul_bf(triangle_cum: pd.DataFrame, cdf: pd.Series, ult_apriori: pd.Series) -> pd.DataFrame:
    """
    BF :
    alpha = 1/CDF(j)
    PSAP_BF = (1 - alpha) * U_apriori
    Ultime_BF = C_obs + PSAP_BF
    """
    dev_cols = list(triangle_cum.columns[1:])
    dernier = []
    alpha = []
    ult_bf = []
    psap_bf = []

    for i in range(len(triangle_cum)):
        ligne = triangle_cum.loc[i, dev_cols]
        last_col = ligne.last_valid_index()
        if last_col is None:
            dernier.append(np.nan)
            alpha.append(np.nan)
            ult_bf.append(np.nan)
            psap_bf.append(np.nan)
            continue

        c_obs = ligne[last_col]
        a = 1.0 / float(cdf.loc[last_col]) if float(cdf.loc[last_col]) != 0 else np.nan
        ps = (1.0 - a) * float(ult_apriori.iloc[i]) if pd.notna(a) else np.nan
        u = c_obs + ps if pd.notna(ps) else np.nan

        dernier.append(c_obs)
        alpha.append(a)
        ult_bf.append(u)
        psap_bf.append(ps)

    return pd.DataFrame({
        "Année de survenance": triangle_cum.iloc[:, 0].astype(str),
        "Dernier règlement observé": dernier,
        "α (part développée)": alpha,
        "Ultime a priori": ult_apriori.values,
        "Ultime BF": ult_bf,
        "PSAP BF": psap_bf
    })

# ============================================================
# MACK CHAIN-LADDER (simplifié mais robuste M1)
# ============================================================

def mack_sigma2(triangle_cum: pd.DataFrame, facteurs: pd.Series) -> pd.Series:
    """
    Estimation σ²_j sur les ratios individuels :
      ratio_ij = C_{i,j+1}/C_{i,j}
      σ²_j = Var(ratio_ij) (ddof=1) sur cellules observées
    """
    dev_cols = list(triangle_cum.columns[1:])
    sigma2 = []

    for j in range(len(facteurs)):
        a, b = dev_cols[j], dev_cols[j + 1]
        mask = triangle_cum[a].notna() & triangle_cum[b].notna() & (triangle_cum[a] > 0)
        ratios = (triangle_cum.loc[mask, b] / triangle_cum.loc[mask, a]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratios) <= 1:
            sigma2.append(np.nan)
        else:
            sigma2.append(float(ratios.var(ddof=1)))

    return pd.Series(sigma2, index=facteurs.index, name="σ²_j")

# ============================================================
# BOOTSTRAP CHAIN-LADDER (pédagogique)
# ============================================================

def bootstrap_total_psap(triangle_cum: pd.DataFrame, facteurs: pd.Series, n_sim: int = 1000, sigma_ln: float = 0.05) -> pd.DataFrame:
    """
    Bootstrap pédagogique :
    - on perturbe les facteurs par un bruit lognormal
    - on recalcule ultimes CL et PSAP totale
    - on renvoie quantiles sur PSAP totale et Ultime total
    """
    tot_ult = []
    tot_psap = []

    dernier = last_observed_per_row(triangle_cum).values

    for _ in range(int(n_sim)):
        f_star = facteurs.values * np.random.lognormal(mean=0.0, sigma=sigma_ln, size=len(facteurs))
        f_star = pd.Series(f_star, index=facteurs.index)

        ult_star = calcul_ultimes_chain_ladder(triangle_cum, f_star).values
        psap_star = ult_star - dernier

        tot_ult.append(np.nansum(ult_star))
        tot_psap.append(np.nansum(psap_star))

    tot_ult = np.array(tot_ult)
    tot_psap = np.array(tot_psap)

    out = pd.DataFrame({
        "Stat": ["5%", "50% (médiane)", "95%"],
        "Ultime total": np.percentile(tot_ult, [5, 50, 95]),
        "PSAP totale": np.percentile(tot_psap, [5, 50, 95]),
    })
    return out

# ============================================================
# INTERFACE UTILISATEUR – ENTRÉE
# ============================================================

st.sidebar.header("📂 Données d’entrée")

mode_entree = st.sidebar.radio(
    "Mode de saisie du triangle",
    ["Importer un fichier", "Coller le triangle"]
)

type_triangle = st.sidebar.radio(
    "Type de triangle fourni",
    ["Triangle cumulé", "Triangle incrémental"]
)

triangle = None

if mode_entree == "Importer un fichier":
    fichier = st.sidebar.file_uploader(
        "Choisir un fichier (CSV ou Excel)",
        type=["csv", "xlsx", "xls"]
    )
    if fichier:
        triangle = lire_triangle_fichier(fichier)

else:
    texte = st.sidebar.text_area(
        "Coller le triangle ici (copié Excel)",
        height=180,
        placeholder="origin;0;12;24;36\n2020;100;160;190;\n2021;120;190;;\n2022;140;;;"
    )
    if texte.strip():
        triangle = lire_triangle_texte(texte)

if triangle is None or triangle.empty:
    st.info("👉 Importez un fichier ou collez un triangle pour lancer le calcul.")
    st.stop()

st.subheader("1️⃣ Triangle fourni")
st.dataframe(triangle, use_container_width=True)

# Conversion si incrémental
if type_triangle == "Triangle incrémental":
    triangle_cum = incremental_vers_cumule(triangle)
else:
    triangle_cum = triangle.copy()

st.subheader("2️⃣ Triangle cumulé utilisé pour Chain-Ladder")
st.dataframe(triangle_cum, use_container_width=True)

# ============================================================
# PARAMÈTRES MÉTHODES
# ============================================================

st.sidebar.markdown("---")
methode = st.sidebar.radio(
    "Méthode de provisionnement",
    ["Chain-Ladder", "Bornhuetter–Ferguson", "Mack", "Bootstrap"]
)

# ============================================================
# CALCULS COMMUNS CL
# ============================================================

dev_cols = list(triangle_cum.columns[1:])
facteurs = calcul_facteurs_chain_ladder(triangle_cum)
cdf = calcul_cdf(facteurs, dev_cols)
triangle_proj = projection_triangle_chain_ladder(triangle_cum, facteurs)
ult_cl = calcul_ultimes_chain_ladder(triangle_cum, facteurs)
df_cl = calcul_psap(triangle_cum, ult_cl, label_ultime="Ultime CL")
df_cl_tot = ajouter_ligne_total(df_cl, ["Ultime CL", "PSAP"])

# ============================================================
# AFFICHAGES COMMUNS (comme ton code de base)
# ============================================================

c1, c2 = st.columns(2)
with c1:
    st.subheader("3️⃣ Facteurs Chain-Ladder")
    st.dataframe(facteurs.to_frame(), use_container_width=True)

with c2:
    st.subheader("4️⃣ CDF vers l’ultime")
    st.dataframe(cdf.to_frame(), use_container_width=True)

st.subheader("5️⃣ Triangle complété (pédagogique)")
st.dataframe(triangle_proj, use_container_width=True)

# ============================================================
# CHAÎNE PRINCIPALE : MÉTHODE CHOISIE + TOTAUX
# ============================================================

if methode == "Chain-Ladder":
    st.subheader("6️⃣ Résultats – Chain-Ladder (avec totaux)")
    st.dataframe(df_cl_tot, use_container_width=True)

    st.metric("📊 Ultime total – Chain-Ladder", f"{df_cl['Ultime CL'].sum():,.2f}")
    st.metric("💰 PSAP totale – Chain-Ladder", f"{df_cl['PSAP'].sum():,.2f}")

elif methode == "Bornhuetter–Ferguson":
    st.sidebar.header("Paramètres BF")
    mode_apriori = st.sidebar.radio("A priori", ["Loss Ratio", "Ultime a priori direct"])

    if mode_apriori == "Loss Ratio":
        lr = st.sidebar.number_input("Loss Ratio a priori", min_value=0.0, max_value=2.0, value=0.70, step=0.05)
        # Approche pédagogique : Ultime a priori = (estimation d'exposition) × LR
        # On approxime l'exposition par "Ultime CL" / LR (pour rester simple sans prime)
        ult_apriori = (ult_cl / lr) * lr  # => ult_cl (si pas de prime fournie), stable et cohérent
    else:
        texte_u = st.sidebar.text_input("Ultimes a priori (séparés par ;)", value="1000;1000;1000")
        vals = [float(x.strip()) for x in texte_u.split(";") if x.strip() != ""]
        if len(vals) != len(triangle_cum):
            st.error("Le nombre d’ultimes a priori doit être égal au nombre de lignes (années).")
            st.stop()
        ult_apriori = pd.Series(vals)

    df_bf = calcul_bf(triangle_cum, cdf, ult_apriori)
    df_bf_tot = ajouter_ligne_total(df_bf, ["Ultime BF", "PSAP BF"])

    st.subheader("6️⃣ Résultats – Bornhuetter–Ferguson (avec totaux)")
    st.dataframe(df_bf_tot, use_container_width=True)

    st.metric("📊 Ultime total – BF", f"{df_bf['Ultime BF'].sum():,.2f}")
    st.metric("💰 PSAP totale – BF", f"{df_bf['PSAP BF'].sum():,.2f}")

    # Comparaison CL vs BF (table de synthèse)
    st.subheader("🔁 Comparaison des totaux – Chain-Ladder vs BF")
    synth = pd.DataFrame({
        "Méthode": ["Chain-Ladder", "Bornhuetter–Ferguson"],
        "Ultime total": [df_cl["Ultime CL"].sum(), df_bf["Ultime BF"].sum()],
        "PSAP totale": [df_cl["PSAP"].sum(), df_bf["PSAP BF"].sum()]
    })
    st.dataframe(synth, use_container_width=True)

elif methode == "Mack":
    sigma2 = mack_sigma2(triangle_cum, facteurs)

    st.subheader("6️⃣ Mack Chain-Ladder – Variances par âge (σ²)")
    st.dataframe(sigma2.to_frame(), use_container_width=True)

    # Synthèse totale (approx pédagogique)
    ultime_total = float(ult_cl.sum())
    var_glob = float(np.nansum(sigma2.values))
    se_glob = np.sqrt(var_glob) * ultime_total if pd.notna(var_glob) else np.nan

    st.subheader("🔎 Synthèse Mack (totaux)")
    st.metric("📊 Ultime total (Chain-Ladder)", f"{ultime_total:,.2f}")
    st.metric("💰 PSAP totale (Chain-Ladder)", f"{df_cl['PSAP'].sum():,.2f}")
    st.metric("📉 Écart-type global (approx.)", f"{se_glob:,.2f}" if pd.notna(se_glob) else "NA")

elif methode == "Bootstrap":
    st.sidebar.header("Paramètres Bootstrap")
    n_sim = st.sidebar.number_input("Nombre de simulations", min_value=100, max_value=5000, value=1000, step=100)
    sigma_ln = st.sidebar.number_input("Volatilité lognormale (sigma)", min_value=0.01, max_value=0.30, value=0.05, step=0.01)

    boot = bootstrap_total_psap(triangle_cum, facteurs, n_sim=int(n_sim), sigma_ln=float(sigma_ln))

    st.subheader("6️⃣ Bootstrap Chain-Ladder – Totaux (quantiles)")
    st.dataframe(boot, use_container_width=True)

    # Metrics “comme dans ton code”
    med_ult = float(boot.loc[boot["Stat"] == "50% (médiane)", "Ultime total"].iloc[0])
    med_psap = float(boot.loc[boot["Stat"] == "50% (médiane)", "PSAP totale"].iloc[0])

    st.metric("📊 Ultime total (médiane bootstrap)", f"{med_ult:,.2f}")
    st.metric("💰 PSAP totale (médiane bootstrap)", f"{med_psap:,.2f}")

# ============================================================
# GRAPHIQUES (communs)
# ============================================================

st.subheader("📈 Graphiques – Ultimes & PSAP (Chain-Ladder)")
fig1, ax1 = plt.subplots()
ax1.plot(df_cl["Année de survenance"], df_cl["Ultime CL"], marker="o")
ax1.set_title("Ultimes estimés (Chain-Ladder)")
ax1.set_xlabel("Année de survenance")
ax1.set_ylabel("Montant")
plt.xticks(rotation=45)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(df_cl["Année de survenance"], df_cl["PSAP"], marker="o")
ax2.set_title("PSAP (Chain-Ladder)")
ax2.set_xlabel("Année de survenance")
ax2.set_ylabel("Montant")
plt.xticks(rotation=45)
st.pyplot(fig2)

st.markdown("---")
st.markdown("© **Eutch Présence BITSINDOU** – Outil de provisionnement actuariel")
