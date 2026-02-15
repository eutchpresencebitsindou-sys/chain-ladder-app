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
# IMPORT / PARSING / TRANSFORMATIONS
# =========================

def lire_triangle_fichier(fichier) -> pd.DataFrame:
    """Lit CSV / XLSX et nettoie les colonnes (première colonne = origine)."""
    nom = fichier.name.lower()
    if nom.endswith(".csv"):
        df = pd.read_csv(fichier, sep=None, engine="python")
    else:
        df = pd.read_excel(fichier)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def lire_triangle_texte(texte: str, sep_hint: str = None) -> pd.DataFrame:
    """Parse un triangle collé (séparateurs : tab, ;, ,, espace)."""
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
    df = pd.read_csv(io.StringIO(txt), sep=sep, engine="python")
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def incremental_vers_cumule(triangle_inc: pd.DataFrame) -> pd.DataFrame:
    """
    Conversion incrémental → cumulé
    Règle : cumuler jusqu'au premier NA, ensuite NA (pas de cumul à travers les vides).
    """
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
    """Convertit un triangle cumulé en incrémental (différence horizontale), en gardant NA à droite."""
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

# =========================
# CHAIN-LADDER (facteurs & projections)
# =========================

def compute_link_factors(tri_cum: pd.DataFrame, method: str = "volume_weighted", last_n: int = None) -> pd.Series:
    """
    Calcule les facteurs âge->âge.
    method: "volume_weighted", "simple_average", "median", "latest"
    last_n: si non None et >0, on prend seulement les last_n origines (les plus récentes)
    """
    dev_cols = list(tri_cum.columns[1:])
    n_dev = len(dev_cols)
    factors = []
    for j in range(n_dev - 1):
        a = dev_cols[j]
        b = dev_cols[j + 1]
        mask = tri_cum[a].notna() & tri_cum[b].notna() & (tri_cum[a] > 0)
        tmp = tri_cum.loc[mask, [a, b]].copy()
        if tmp.empty:
            factors.append(np.nan)
            continue
        if last_n is not None and last_n > 0 and len(tmp) > last_n:
            tmp = tmp.tail(last_n)
        if method == "volume_weighted":
            f = tmp[b].sum() / tmp[a].sum()
        elif method == "simple_average":
            f = (tmp[b] / tmp[a]).mean()
        elif method == "median":
            f = (tmp[b] / tmp[a]).median()
        elif method == "latest":
            f = (tmp[b].iloc[-1] / tmp[a].iloc[-1])
        else:
            raise ValueError("Méthode inconnue")
        factors.append(float(f))
    return pd.Series(factors, index=[f"{dev_cols[j]}->{dev_cols[j+1]}" for j in range(n_dev - 1)], name="Facteur")

def compute_cdf_from_factors(factors: pd.Series, dev_cols: list[str]) -> pd.Series:
    """CDF vers l'ultime alignée sur dev_cols (dernier dev -> CDF=1)."""
    n = len(dev_cols)
    cdf = np.ones(n)
    f = factors.values.astype(float)
    for j in range(n - 2, -1, -1):
        cdf[j] = cdf[j + 1] * f[j]
    return pd.Series(cdf, index=dev_cols, name="CDF")

def project_triangle_using_factors(tri_cum: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
    """Projette cumulativement le triangle en complétant les NA quand possible (séquentiel)."""
    tri = tri_cum.copy()
    dev_cols = list(tri.columns[1:])
    for i in range(len(tri)):
        for j in range(len(dev_cols) - 1):
            a = dev_cols[j]
            b = dev_cols[j + 1]
            if pd.notna(tri.loc[i, a]) and pd.isna(tri.loc[i, b]):
                tri.loc[i, b] = tri.loc[i, a] * float(factors.iloc[j])
    return tri

def ultimates_from_factors(tri_cum: pd.DataFrame, factors: pd.Series) -> pd.Series:
    """Calcul des ultimes: dernier cumul observé × produit des facteurs restants."""
    dev_cols = list(tri_cum.columns[1:])
    ult = []
    for i in range(len(tri_cum)):
        row = tri_cum.loc[i, dev_cols]
        last = row.last_valid_index()
        if last is None:
            ult.append(np.nan)
        else:
            pos = dev_cols.index(last)
            cumul = float(row[last])
            prod = 1.0
            if pos < len(factors):
                prod = float(np.prod(factors.values[pos:]))
            ult.append(cumul * prod)
    return pd.Series(ult, name="Ultime")

def psap_table(tri_cum: pd.DataFrame, ult_series: pd.Series, label_ult: str) -> pd.DataFrame:
    """Compose un DataFrame avec Année, dernier cumul, ultime et PSAP."""
    dernier = dernier_cumule_par_ligne(tri_cum)
    df = pd.DataFrame({
        "Année de survenance": tri_cum.iloc[:, 0].astype(str),
        "Dernier cumul observé": dernier.values,
        label_ult: ult_series.values
    })
    df["PSAP"] = df[label_ult] - df["Dernier cumul observé"]
    return df

# =========================
# LONDON-CHAIN (Benjamin & Eagles) — Régression affine âge-à-âge
# =========================

def london_chain_fit(tri_cum: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Estime λ_j et β_j par MCO sur chaque transition j -> j+1 :
      C_{i,j+1} = λ_j * C_{i,j} + β_j
    """
    dev_cols = list(tri_cum.columns[1:])
    J = len(dev_cols)

    rows = []
    lambdas = []
    betas = []
    idx = []

    for j in range(J - 1):
        a = dev_cols[j]
        b = dev_cols[j + 1]
        valid = tri_cum[[a, b]].dropna()
        x = valid[a].astype(float).values
        y = valid[b].astype(float).values
        nobs = len(valid)

        if nobs == 0:
            lam = np.nan
            beta = np.nan
            r2 = np.nan
            Cbar_x = np.nan
            Cbar_y = np.nan
        elif nobs == 1:
            Cbar_x = float(x.mean())
            Cbar_y = float(y.mean())
            lam = (Cbar_y / Cbar_x) if Cbar_x != 0 else np.nan
            beta = 0.0
            r2 = np.nan
        else:
            Cbar_x = float(x.mean())
            Cbar_y = float(y.mean())
            num = float((x * y).sum() - nobs * Cbar_x * Cbar_y)
            den = float((x * x).sum() - nobs * (Cbar_x ** 2))
            if abs(den) < 1e-12:
                lam = (Cbar_y / Cbar_x) if Cbar_x != 0 else np.nan
                beta = 0.0
                r2 = np.nan
            else:
                lam = num / den
                beta = Cbar_y - lam * Cbar_x
                y_hat = lam * x + beta
                sst = float(((y - Cbar_y) ** 2).sum())
                sse = float(((y - y_hat) ** 2).sum())
                r2 = 1.0 - (sse / sst) if sst > 0 else np.nan

        rows.append({
            "Transition": f"{a} → {b}",
            "n_obs": nobs,
            "C̄_j": Cbar_x,
            "C̄_{j+1}": Cbar_y,
            "λ_j (London)": lam,
            "β_j (London)": beta,
            "R²": r2
        })
        lambdas.append(lam)
        betas.append(beta)
        idx.append(a)

    table_fit = pd.DataFrame(rows)
    lambdas = pd.Series(lambdas, index=idx, name="lambda_j")
    betas = pd.Series(betas, index=idx, name="beta_j")
    return table_fit, lambdas, betas

def london_chain_project(tri_cum: pd.DataFrame, lambdas: pd.Series, betas: pd.Series) -> pd.DataFrame:
    """Complète le triangle cumulé avec C_{i,j+1} = λ_j*C_{i,j} + β_j (séquentiel)."""
    tri = tri_cum.copy()
    dev_cols = list(tri.columns[1:])
    J = len(dev_cols)
    for i in range(len(tri)):
        for j in range(J - 1):
            a = dev_cols[j]
            b = dev_cols[j + 1]
            if pd.notna(tri.loc[i, a]) and pd.isna(tri.loc[i, b]):
                lam = float(lambdas.loc[a]) if (a in lambdas.index and pd.notna(lambdas.loc[a])) else np.nan
                beta = float(betas.loc[a]) if (a in betas.index and pd.notna(betas.loc[a])) else np.nan
                if pd.notna(lam) and pd.notna(beta):
                    tri.loc[i, b] = lam * float(tri.loc[i, a]) + beta
    return tri

def london_chain_ultimes_psap(tri_cum: pd.DataFrame, tri_proj: pd.DataFrame) -> pd.DataFrame:
    """Ultime London = dernière colonne du triangle projeté ; PSAP = Ultime - dernier cumul observé."""
    origin = tri_cum.columns[0]
    dev_cols = list(tri_cum.columns[1:])
    last_dev = dev_cols[-1]
    last_obs = dernier_cumule_par_ligne(tri_cum)
    ult = tri_proj[last_dev].astype(float)
    df = pd.DataFrame({
        "Année de survenance": tri_cum[origin].astype(str),
        "Dernier cumul observé": last_obs.values,
        "Ultime London": ult.values
    })
    df["PSAP London"] = df["Ultime London"] - df["Dernier cumul observé"]
    return df

# =========================
# BORNHUETTER-FERGUSON (IRIAF logic)
# =========================

def bornhuetter_ferguson(tri_cum: pd.DataFrame, cdf: pd.Series, ult_apriori: pd.Series) -> pd.DataFrame:
    """
    BF selon : PSAP_i = (1 - 1/CDF_j) * U_apriori_i
    Ultime_BF = C_{i,j} + PSAP_i
    où j est le dernier développement observé pour la ligne i.
    """
    dev_cols = list(tri_cum.columns[1:])
    dernier = []
    alpha = []
    psap = []
    ultime = []
        row = tri_cum.loc[i, dev_cols]
        last_col = row.last_valid_index()
        if last_col is None:
            dernier.append(np.nan); alpha.append(np.nan); psap.append(np.nan); ultime.append(np.nan)
            continue
        c_obs = float(row[last_col])
        cdf_j = float(cdf.loc[last_col]) if last_col in cdf.index else np.nan
        a = (1.0 / cdf_j) if (not pd.isna(cdf_j) and cdf_j != 0) else np.nan
        ps = (1.0 - a) * float(ult_apriori.iloc[i]) if (not pd.isna(a)) else np.nan
        u = c_obs + ps if (not pd.isna(ps)) else np.nan
        dernier.append(c_obs); alpha.append(a); psap.append(ps); ultime.append(u)
    return pd.DataFrame({
        "Année de survenance": tri_cum.iloc[:, 0].astype(str),
        "Dernier cumul observé": dernier,
        "α (1/CDF)": alpha,
        "Ultime a priori": ult_apriori.values,
        "Ultime BF": ultime,
        "PSAP BF": psap
    })

# =========================
# MACK — estimation σ² et SE par origine (version stable)
# =========================

def mack_sigma2(tri_cum: pd.DataFrame, factors: pd.Series) -> pd.Series:
    """
    Estimation σ²_j (pondérée) :
    σ²_j = sum_i [ C_{i,j} * (r_ij - f_j)^2 ] / (n_j - 1) / sum_i C_{i,j}
    """
    dev_cols = list(tri_cum.columns[1:])
    s2 = []
    for j in range(len(factors)):
        a = dev_cols[j]
        b = dev_cols[j + 1]
        mask = tri_cum[a].notna() & tri_cum[b].notna() & (tri_cum[a] > 0)
        tmp = tri_cum.loc[mask, [a, b]].copy()
        if len(tmp) <= 1:
            s2.append(np.nan); continue
        r = (tmp[b] / tmp[a]).replace([np.inf, -np.inf], np.nan).dropna()
        tmp = tmp.loc[r.index]
        f_j = float(factors.iloc[j])
        num = (tmp[a] * (r - f_j) ** 2).sum()
        den = tmp[a].sum()
        s2.append(float(num / ((len(tmp) - 1) * den)) if den > 0 else np.nan)
    return pd.Series(s2, index=factors.index, name="σ²_j")

def mack_se_by_origin(tri_cum: pd.DataFrame, factors: pd.Series, sigma2: pd.Series) -> pd.Series:
    """
    Var(U_i) ≈ sum_{k=j_i}^{J-1} (C_{i,k}^2 * σ²_k * prod_{m=k+1}^{J-1} f_m^2)
    """
    dev_cols = list(tri_cum.columns[1:])
    tri_proj = project_triangle_using_factors(tri_cum, factors)

    se = []
    f = factors.values.astype(float)

    prod_sq = np.ones(len(dev_cols))
    for k in range(len(dev_cols) - 2, -1, -1):
        prod_sq[k] = prod_sq[k + 1] * (f[k] ** 2)

    for i in range(len(tri_cum)):
        row_obs = tri_cum.loc[i, dev_cols]
        last = row_obs.last_valid_index()
        if last is None:
            se.append(np.nan); continue
        j0 = dev_cols.index(last)
        var = 0.0
        for k in range(j0, len(dev_cols) - 1):
            c_ik = float(tri_proj.loc[i, dev_cols[k]])
            s2k = sigma2.iloc[k] if k < len(sigma2) else np.nan
            if pd.isna(s2k):
                continue
            tail = prod_sq[k + 1] if (k + 1) < len(dev_cols) else 1.0
            var += (c_ik ** 2) * float(s2k) * tail
        se.append(np.sqrt(var) if var >= 0 else np.nan)

    return pd.Series(se, name="SE_Ultime")

# =========================
# BOOTSTRAP ODP (résidus Pearson sur incrémental) — robuste
# =========================

def bootstrap_odp(tri_cum: pd.DataFrame, n_sim: int = 2000, seed: int | None = None):
    rng = np.random.default_rng(seed)
    dev = list(tri_cum.columns[1:])

    base_factors = compute_link_factors(tri_cum, method="volume_weighted", last_n=None)
    tri_proj = project_triangle_using_factors(tri_cum, base_factors)
    inc_obs = cumule_vers_incremental(tri_cum)
    inc_mu = cumule_vers_incremental(tri_proj)

    mask_obs = inc_obs[dev].notna()
    mu = inc_mu[dev].where(mask_obs)
    y = inc_obs[dev].where(mask_obs)
    mu_safe = mu.where(mu > 0)

    resid = (y - mu_safe) / np.sqrt(mu_safe)
    resid_vals = resid.stack().dropna().values
    if resid_vals.size == 0:
        return None

    dernier = dernier_cumule_par_ligne(tri_cum).values
    tot_ult = []
    tot_psap = []
    by_origin = []

    obs_positions = [(i, j) for i in range(len(tri_cum)) for j in range(len(dev)) if mask_obs.iloc[i, j]]
    n_obs = len(obs_positions)

    for _ in range(int(n_sim)):
        r_star = rng.choice(resid_vals, size=n_obs, replace=True)
        y_star = mu.copy()

        for (i, j), r in zip(obs_positions, r_star):
            mu_ij = mu_safe.iat[i, j]
            if pd.isna(mu_ij) or mu_ij <= 0:
                y_star.iat[i, j] = y.iat[i, j] if not pd.isna(y.iat[i, j]) else 0.0
            else:
                val = float(mu_ij + r * np.sqrt(mu_ij))
                y_star.iat[i, j] = max(val, 0.0)

        tri_star = tri_cum.copy()
        for i in range(len(tri_cum)):
            cumul = 0.0
            stop = False
            for j, col in enumerate(dev):
                if not mask_obs.iat[i, j]:
                    tri_star.iat[i, j + 1] = np.nan
                    stop = True
                else:
                    if stop:
                        tri_star.iat[i, j + 1] = np.nan
                    else:
                        cumul += float(y_star.iat[i, j])
                        tri_star.iat[i, j + 1] = cumul

        f_star = compute_link_factors(tri_star, method="volume_weighted", last_n=None)
        ult_star = ultimates_from_factors(tri_star, f_star).values
        psap_star = ult_star - dernier

        tot_ult.append(np.nansum(ult_star))
        tot_psap.append(np.nansum(psap_star))
        by_origin.append(psap_star)

    tot_ult = np.array(tot_ult)
    tot_psap = np.array(tot_psap)
    by_origin = np.vstack(by_origin)

    return {"tot_ult": tot_ult, "tot_psap": tot_psap, "by_origin": by_origin}

# =========================
# SIDEBAR – ENTRÉE / OPTIONS (conserver ton UI)
# =========================

st.sidebar.header("⚙️ Paramètres & Données")

triangle_mode = st.sidebar.radio("Mode de saisie du triangle", ["Importer un fichier", "Coller le triangle"], index=0)
triangle_type = st.sidebar.radio("Type de triangle fourni", ["Triangle cumulé", "Triangle incrémental"], index=0)

df_raw = None
if triangle_mode == "Importer un fichier":
    uploaded = st.sidebar.file_uploader("Importer un triangle (CSV ou Excel)", type=["csv", "xlsx"])
    if uploaded is not None:
        df_raw = lire_triangle_fichier(uploaded)
else:
    paste = st.sidebar.text_area("Coller le triangle (en-têtes)", height=160, placeholder="Ex:\norigin;0;12;24\n2020;100;150;\n2021;120;;")
    sep_choice = st.sidebar.selectbox("Séparateur (si collage)", ["auto", ";", ",", "tab", "space"], index=0)

    def sep_map(x):
        return None if x == "auto" else ("\t" if x == "tab" else (r"\s+" if x == "space" else x))

    if paste.strip():
        df_raw = lire_triangle_texte(paste, sep_hint=sep_map(sep_choice))

if df_raw is None or df_raw.empty:
    st.info("👉 Importez un fichier ou collez un triangle pour commencer.")
    st.stop()

st.subheader("1) Triangle importé")
st.dataframe(df_raw, use_container_width=True)

if triangle_type == "Triangle incrémental":
    tri = incremental_vers_cumule(df_raw)
else:
    tri = df_raw.copy()

st.subheader("2) Triangle cumulé (utilisé pour les méthodes)")
st.dataframe(tri, use_container_width=True)

st.sidebar.markdown("---")
methode = st.sidebar.selectbox(
    "Méthode de provisionnement",
    ["Chain-Ladder", "London-Chain", "Bornhuetter–Ferguson", "Mack", "Bootstrap"],
    index=0
)

selection_method = st.sidebar.selectbox(
    "Sélection des facteurs (Chain-Ladder)",
    ["volume_weighted", "simple_average", "median", "latest"],
    index=0
)
last_n = st.sidebar.number_input("Utiliser seulement les N dernières origines (0 = tout)", min_value=0, max_value=100, value=0, step=1)

# On conserve l'option, même si elle n'est pas utilisée (pas d'impact sur l'UI)
london_decay = st.sidebar.slider("London-Chain : poids decay (0→récent seul, 1→uniforme)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
n_sim = st.sidebar.number_input("Bootstrap : nombre de simulations", min_value=100, max_value=20000, value=2000, step=100)
seed = st.sidebar.number_input("Bootstrap : graine (seed)", min_value=0, max_value=1_000_000, value=42, step=1)

st.sidebar.markdown("---")
show_proj = st.sidebar.checkbox("Afficher triangle complété (pédagogique)", value=True)
show_graphs = st.sidebar.checkbox("Afficher graphiques pour chaque méthode", value=True)

# =========================
# CALCULS COMMUNS
# =========================

dev_cols = list(tri.columns[1:])

factors_cl = compute_link_factors(tri, method=selection_method, last_n=(None if last_n == 0 else int(last_n)))
cdf_cl = compute_cdf_from_factors(factors_cl, dev_cols)
tri_proj_cl = project_triangle_using_factors(tri, factors_cl)
ult_cl = ultimates_from_factors(tri, factors_cl)
df_cl = psap_table(tri, ult_cl, "Ultime CL")
df_cl_tot = ajouter_ligne_total(df_cl, ["Ultime CL", "PSAP"])

# ✅ CORRECTION PRINCIPALE :
# On supprime le "London factors" fictif (fonction london_chain_factors inexistante).
# La méthode London-Chain est calculée UNIQUEMENT dans son bloc dédié avec λ_j et β_j.

# =========================
# AFFICHAGE COMMUNS (ne pas casser ton UI)
# =========================

c1, c2 = st.columns(2)
with c1:
    st.subheader("3) Link ratios / Facteurs (âge → âge) — Chain-Ladder")
    st.dataframe(
        factors_cl.to_frame(name="Facteur").reset_index().rename(columns={"index": "Transition"}),
        use_container_width=True
    )
with c2:
    st.subheader("4) CDF vers l'ultime")
    st.dataframe(
        pd.DataFrame({"dev": dev_cols, "CDF": cdf_cl.values}),
        use_container_width=True
    )

if show_proj:
    st.subheader("5) Triangle complété (Chain-Ladder - pédagogique)")
    st.dataframe(tri_proj_cl, use_container_width=True)

# =========================
# LOGIQUE PAR METHODE (résultats + graphiques + export sheets)
# =========================

sheets = {}

if methode == "Chain-Ladder":
    st.subheader("6) Résultats — Chain-Ladder")
    st.dataframe(df_cl_tot, use_container_width=True)
    st.metric("IBNR total (CL)", f"{df_cl['PSAP'].sum():,.2f}")

    sheets = {
        "triangle_cumule": tri,
        "triangle_projete_CL": tri_proj_cl,
        "facteurs_CL": factors_cl.to_frame(),
        "cdf_CL": cdf_cl.to_frame(),
        "ultimes_psap_CL": df_cl_tot
    }

    if show_graphs:
        st.subheader("📈 Graphiques — Chain-Ladder")
        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots()
            ax.plot(df_cl["Année de survenance"], df_cl["Ultime CL"], marker="o")
            ax.set_title("Ultimes par année (Chain-Ladder)")
            ax.set_xlabel("Année")
            ax.set_ylabel("Ultime")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        with g2:
            fig, ax = plt.subplots()
            ax.bar(df_cl["Année de survenance"], df_cl["PSAP"])
            ax.set_title("PSAP par année (Chain-Ladder)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

elif methode == "London-Chain":
    st.subheader("6) Résultats — London-Chain (affine)")

    table_fit, lambdas, betas = london_chain_fit(tri)

    st.subheader("Paramètres London (λ_j, β_j) + diagnostic d'alignement")
    st.dataframe(table_fit, use_container_width=True)

    tri_london_proj = london_chain_project(tri, lambdas, betas)

    st.subheader("Triangle cumulé complété (London-Chain)")
    st.dataframe(tri_london_proj, use_container_width=True)

    df_london = london_chain_ultimes_psap(tri, tri_london_proj)
    df_london_tot = ajouter_ligne_total(df_london, ["Ultime London", "PSAP London"])

    st.subheader("Ultimes & PSAP — London-Chain")
    st.dataframe(df_london_tot, use_container_width=True)

    a1, a2 = st.columns(2)
    with a1:
        st.metric("Ultime total (London)", f"{df_london['Ultime London'].sum():,.2f}")
    with a2:
        st.metric("PSAP totale (London)", f"{df_london['PSAP London'].sum():,.2f}")

    if show_graphs:
        st.subheader("📈 Graphiques — London-Chain")
        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots()
            ax.plot(df_london["Année de survenance"], df_london["Ultime London"], marker="o")
            ax.set_title("Ultimes — London-Chain")
            ax.set_xlabel("Année de survenance")
            ax.set_ylabel("Montant")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        with g2:
            fig, ax = plt.subplots()
            ax.plot(df_london["Année de survenance"], df_london["PSAP London"], marker="o")
            ax.set_title("PSAP — London-Chain")
            ax.set_xlabel("Année de survenance")
            ax.set_ylabel("Montant")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    sheets = {
        "triangle_cumule": tri,
        "london_parametres": table_fit,
        "triangle_london_projete": tri_london_proj,
        "resultats_London": df_london_tot
    }

elif methode == "Bornhuetter–Ferguson":
    st.sidebar.markdown("### Paramètres BF")
    bf_mode = st.sidebar.radio("A priori BF", ["Ultime a priori direct", "Prime & Loss Ratio"], index=0)

    if bf_mode == "Ultime a priori direct":
        txt_u = st.sidebar.text_area("Ultimes a priori (séparées par ;)", value=";".join(["1000"] * len(tri)), height=80)
        vals = [v.strip() for v in txt_u.split(";") if v.strip() != ""]
        if len(vals) != len(tri):
            st.error("Le nombre d'ultimes a priori doit correspondre au nombre de lignes.")
            st.stop()
        ult_apriori = pd.Series([float(v) for v in vals])
    else:
        lr = st.sidebar.number_input("Loss Ratio a priori", min_value=0.0, max_value=5.0, value=0.7, step=0.01)
        primes_txt = st.sidebar.text_area("Primes par année (séparées par ;)", value=";".join(["10000"] * len(tri)), height=80)
        pvals = [v.strip() for v in primes_txt.split(";") if v.strip() != ""]
        if len(pvals) != len(tri):
            st.error("Le nombre de primes doit correspondre au nombre de lignes.")
            st.stop()
        primes = pd.Series([float(v) for v in pvals])
        ult_apriori = primes * float(lr)

    df_bf = bornhuetter_ferguson(tri, cdf_cl, ult_apriori)
    df_bf_tot = ajouter_ligne_total(df_bf, ["Ultime BF", "PSAP BF"])

    st.subheader("6) Résultats — Bornhuetter–Ferguson")
    st.dataframe(df_bf_tot, use_container_width=True)
    st.metric("IBNR total (BF)", f"{df_bf['PSAP BF'].sum():,.2f}")

    sheets = {
        "triangle_cumule": tri,
        "ultimes_apriori": pd.DataFrame({"Ultime_apriori": ult_apriori}),
        "resultats_BF": df_bf_tot,
        "resultats_CL": df_cl_tot
    }

    if show_graphs:
        st.subheader("📈 Graphiques — BF vs CL")
        fig, ax = plt.subplots()
        ax.plot(df_cl["Année de survenance"], df_cl["Ultime CL"], marker="o", label="CL Ultime")
        ax.plot(df_bf["Année de survenance"], df_bf["Ultime BF"], marker="o", label="BF Ultime")
        ax.set_title("Comparaison Ultimes CL vs BF")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif methode == "Mack":
    st.subheader("6) Résultats — Mack (variances & IC)")

    sigma2 = mack_sigma2(tri, factors_cl)
    se_by_origin = mack_se_by_origin(tri, factors_cl, sigma2)

    df_mack = df_cl.copy()
    df_mack["SE_Ultime"] = se_by_origin.values

    alpha = st.sidebar.selectbox("Niveau d'IC (Mack)", ["90%", "95%", "99%"], index=1)
    z = {"90%": 1.645, "95%": 1.96, "99%": 2.576}[alpha]

    df_mack["IC bas"] = df_mack["Ultime CL"] - z * df_mack["SE_Ultime"]
    df_mack["IC haut"] = df_mack["Ultime CL"] + z * df_mack["SE_Ultime"]
    df_mack["PSAP"] = df_mack["Ultime CL"] - df_mack["Dernier cumul observé"]

    df_mack_tot = ajouter_ligne_total(df_mack, ["Ultime CL", "PSAP"])
    st.dataframe(df_mack_tot, use_container_width=True)

    ultime_total = df_mack["Ultime CL"].sum()
    se_total = float(np.sqrt(np.nansum(df_mack["SE_Ultime"].dropna().values ** 2))) if df_mack["SE_Ultime"].notna().any() else np.nan
    ic_global_bas = ultime_total - z * se_total if not np.isnan(se_total) else np.nan
    ic_global_haut = ultime_total + z * se_total if not np.isnan(se_total) else np.nan

    st.subheader("📐 Intervalle de confiance – Ensemble des années (Mack)")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.metric("Ultime total", f"{ultime_total:,.2f}")
    with cc2:
        st.metric("Écart-type total", f"{se_total:,.2f}" if not np.isnan(se_total) else "NA")
    with cc3:
        st.metric(f"IC global {alpha}", f"[{ic_global_bas:,.2f} ; {ic_global_haut:,.2f}]" if not np.isnan(se_total) else "NA")

    sheets = {
        "triangle_cumule": tri,
        "facteurs_CL": factors_cl.to_frame(),
        "sigma2_par_age": sigma2.to_frame(),
        "resultats_Mack": df_mack_tot
    }

    if show_graphs:
        st.subheader("📈 Graphiques — Mack")
        fig, ax = plt.subplots()
        ax.errorbar(
            df_mack["Année de survenance"],
            df_mack["Ultime CL"],
            yerr=z * df_mack["SE_Ultime"],
            fmt="o",
            capsize=4
        )
        ax.set_title("Ultimes (Mack) avec intervalles de confiance")
        ax.set_xlabel("Année de survenance")
        ax.set_ylabel("Montant")
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif methode == "Bootstrap":
    st.subheader("6) Résultats — Bootstrap (ODP)")
    res_boot = bootstrap_odp(tri, n_sim=int(n_sim), seed=int(seed))

    if res_boot is None:
        st.warning("Bootstrap impossible : pas assez de cellules observées pour construire des résidus.")
        sheets = {"triangle_cumule": tri}
    else:
        tot_ult = res_boot["tot_ult"]
        tot_psap = res_boot["tot_psap"]
        by_origin = res_boot["by_origin"]

        boot_summary = pd.DataFrame({
            "Stat": ["5%", "50% (médiane)", "95%"],
            "Ultime total": np.percentile(tot_ult, [5, 50, 95]),
            "PSAP totale": np.percentile(tot_psap, [5, 50, 95])
        })
        st.dataframe(boot_summary, use_container_width=True)
        st.metric("PSAP totale (médiane bootstrap)", f"{np.percentile(tot_psap, 50):,.2f}")

        years = tri.iloc[:, 0].astype(str).values
        q5 = np.percentile(by_origin, 5, axis=0)
        q50 = np.percentile(by_origin, 50, axis=0)
        q95 = np.percentile(by_origin, 95, axis=0)
        df_boot_by_year = pd.DataFrame({
            "Année de survenance": years,
            "PSAP 5%": q5,
            "PSAP médiane": q50,
            "PSAP 95%": q95
        })
        df_boot_by_year_tot = ajouter_ligne_total(df_boot_by_year, ["PSAP 5%", "PSAP médiane", "PSAP 95%"])

        st.subheader("🔎 Bootstrap — PSAP par année (quantiles)")
        st.dataframe(df_boot_by_year_tot, use_container_width=True)

        sheets = {
            "triangle_cumule": tri,
            "bootstrap_totaux": boot_summary,
            "bootstrap_par_annee": df_boot_by_year_tot,
            "resultats_CL": df_cl_tot
        }

        if show_graphs:
            st.subheader("📈 Graphiques — Bootstrap")
            fig, ax = plt.subplots()
            ax.hist(tot_psap, bins=40)
            ax.set_title("Distribution PSAP totale (bootstrap)")
            st.pyplot(fig)

# =========================
# EXPORT EXCEL (TOUTES MÉTHODES)
# =========================

st.markdown("---")
st.subheader("📥 Télécharger les résultats")

if sheets:
    excel_bytes = to_excel_bytes(sheets)
    st.download_button(
        label="📄 Télécharger les résultats (Excel multi-feuilles)",
        data=excel_bytes,
        file_name="provisionnement_resultats.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Aucun résultat à exporter (vérifie les données).")

st.caption("© Eutch Présence BITSINDOU — Outil de provisionnement actuariel")
