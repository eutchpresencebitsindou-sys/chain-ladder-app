import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

APP_AUTHOR = "Eutch Présence BITSINDOU"

st.set_page_config(
    page_title=f"Provisionnement Non-Vie | {APP_AUTHOR}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Style
# -------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f7fbff 0%, #eef5ff 100%);
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f3d91 0%, #1a73e8 55%, #49a3ff 100%);
        color: white;
        box-shadow: 0 16px 32px rgba(15,61,145,0.18);
        margin-bottom: 1rem;
    }
    .soft-card {
        background: white;
        border: 1px solid rgba(26,115,232,0.12);
        border-radius: 18px;
        padding: 1rem 1rem 0.6rem 1rem;
        box-shadow: 0 8px 24px rgba(16,38,84,0.06);
        margin-bottom: 1rem;
    }
    .mini-kpi {
        background: white;
        border-radius: 18px;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(26,115,232,0.10);
        box-shadow: 0 8px 24px rgba(16,38,84,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 16px;
        background: #eef4ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
def format_num(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:,.2f}".replace(",", " ")


def to_incremental(cum_triangle: pd.DataFrame) -> pd.DataFrame:
    inc = cum_triangle.copy().astype(float)
    for i in range(inc.shape[0]):
        row = inc.iloc[i].to_numpy(dtype=float)
        prev = np.nan
        out = []
        for val in row:
            if np.isnan(val):
                out.append(np.nan)
            else:
                if np.isnan(prev):
                    out.append(val)
                else:
                    out.append(val - prev)
                prev = val
        inc.iloc[i] = out
    return inc


def to_cumulative(inc_triangle: pd.DataFrame) -> pd.DataFrame:
    cum = inc_triangle.copy().astype(float)
    for i in range(cum.shape[0]):
        row = cum.iloc[i].to_numpy(dtype=float)
        total = 0.0
        started_na = False
        out = []
        for val in row:
            if np.isnan(val):
                started_na = True
                out.append(np.nan)
            else:
                if started_na:
                    # On n'accumule pas après le premier NA observé
                    out.append(np.nan)
                else:
                    total += val
                    out.append(total)
        cum.iloc[i] = out
    return cum


def last_observed_by_row(triangle: pd.DataFrame) -> pd.Series:
    vals = []
    for _, row in triangle.iterrows():
        notna = row.dropna()
        vals.append(np.nan if notna.empty else float(notna.iloc[-1]))
    return pd.Series(vals, index=triangle.index)


def observed_dev_index_by_row(triangle: pd.DataFrame) -> List[int]:
    idxs = []
    for _, row in triangle.iterrows():
        arr = row.to_numpy(dtype=float)
        valid = np.where(~np.isnan(arr))[0]
        idxs.append(int(valid[-1]) if len(valid) else -1)
    return idxs


def parse_triangle_from_text(text: str, sep: Optional[str] = None) -> pd.DataFrame:
    text = text.strip()
    if not text:
        raise ValueError("Aucune donnée collée.")
    if sep is None:
        sep = "\t" if "\t" in text else None
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    df = sanitize_triangle(df)
    return df


def sanitize_triangle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.shape[1] < 2:
        raise ValueError("Le triangle doit avoir au moins une colonne d'origine et une colonne de développement.")

    first_col = df.columns[0]
    df[first_col] = df[first_col].astype(str)
    df = df.set_index(first_col)

    new_cols = []
    for c in df.columns:
        try:
            cc = float(str(c).replace(",", "."))
            if cc.is_integer():
                new_cols.append(int(cc))
            else:
                new_cols.append(cc)
        except Exception:
            new_cols.append(str(c))
    df.columns = new_cols

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_index()
    return df


def triangle_mask_lower(triangle: pd.DataFrame) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=triangle.index, columns=triangle.columns)
    obs = observed_dev_index_by_row(triangle)
    for i, last_j in enumerate(obs):
        if last_j + 1 < triangle.shape[1]:
            mask.iloc[i, last_j + 1 :] = True
    return mask


def compute_link_ratios(cum_triangle: pd.DataFrame) -> pd.DataFrame:
    ratios = pd.DataFrame(index=cum_triangle.index)
    cols = list(cum_triangle.columns)
    for j in range(len(cols) - 1):
        c0, c1 = cols[j], cols[j + 1]
        vals = cum_triangle[c1] / cum_triangle[c0]
        vals[(cum_triangle[c0] <= 0) | cum_triangle[c0].isna() | cum_triangle[c1].isna()] = np.nan
        ratios[f"{c0}->{c1}"] = vals
    return ratios


def selected_factors_from_ratios(cum_triangle: pd.DataFrame, mode: str = "pondérée", tail_factor: float = 1.0) -> pd.Series:
    cols = list(cum_triangle.columns)
    factors = []
    for j in range(len(cols) - 1):
        x = cum_triangle.iloc[:, j]
        y = cum_triangle.iloc[:, j + 1]
        valid = x.notna() & y.notna() & (x > 0)
        if valid.sum() == 0:
            factors.append(np.nan)
            continue
        if mode == "simple":
            f = float((y[valid] / x[valid]).mean())
        elif mode == "min":
            f = float((y[valid] / x[valid]).min())
        elif mode == "max":
            f = float((y[valid] / x[valid]).max())
        else:
            f = float(y[valid].sum() / x[valid].sum())
        factors.append(f)
    factors.append(float(tail_factor))
    names = [f"f_{cols[j]}->{cols[j+1]}" for j in range(len(cols) - 1)] + ["tail"]
    return pd.Series(factors, index=names)



def normalize_factors_for_triangle(factors_without_tail: List[float], ncols: int) -> List[float]:
    factors = []
    target = max(ncols - 1, 0)
    raw = list(factors_without_tail) if factors_without_tail is not None else []
    for x in raw[:target]:
        try:
            val = float(np.asarray(x).reshape(-1)[0])
            if not np.isfinite(val) or val <= 0:
                val = 1.0
        except Exception:
            val = 1.0
        factors.append(val)
    if len(factors) < target:
        factors.extend([1.0] * (target - len(factors)))
    return factors

def cdf_from_factors(factors_without_tail: List[float], tail_factor: float = 1.0) -> np.ndarray:
    factors = []
    raw = list(factors_without_tail) if factors_without_tail is not None else []
    for x in raw:
        try:
            val = float(np.asarray(x).reshape(-1)[0])
            if not np.isfinite(val) or val <= 0:
                val = 1.0
        except Exception:
            val = 1.0
        factors.append(val)
    try:
        tail = float(np.asarray(tail_factor).reshape(-1)[0])
        if not np.isfinite(tail) or tail <= 0:
            tail = 1.0
    except Exception:
        tail = 1.0
    all_f = factors + [tail]
    m = len(all_f)
    cdf = np.ones(m, dtype=float)
    prod = 1.0
    for k in range(m - 1, -1, -1):
        prod *= float(all_f[k])
        cdf[k] = prod
    return cdf

def complete_triangle_chain_ladder(cum_triangle: pd.DataFrame, factors_without_tail: List[float]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    tri = cum_triangle.copy().astype(float)
    nrows, ncols = tri.shape
    factors = normalize_factors_for_triangle(factors_without_tail, ncols)

    for i in range(nrows):
        valid_cols = np.where(~tri.iloc[i].isna().to_numpy())[0]
        if len(valid_cols) == 0:
            continue
        last = int(valid_cols[-1])
        last_obs_val = float(tri.iloc[i, last])
        for j in range(last + 1, ncols):
            prev = tri.iat[i, j - 1]
            try:
                prev = float(prev)
            except Exception:
                prev = np.nan
            if np.isnan(prev):
                prev = last_obs_val
            factor = float(factors[j - 1]) if (j - 1) < len(factors) else 1.0
            tri.iat[i, j] = float(prev * factor)

    latest = last_observed_by_row(cum_triangle)
    ultimate = tri.iloc[:, -1].astype(float)
    reserve = (ultimate - latest).astype(float)
    return tri, latest, ultimate, reserve

def dev_cdf_by_row(cum_triangle: pd.DataFrame, factors_without_tail: List[float], tail_factor: float = 1.0) -> pd.Series:
    cdf = cdf_from_factors(factors_without_tail, tail_factor)
    obs = observed_dev_index_by_row(cum_triangle)
    vals = []
    for idx in obs:
        if idx == -1:
            vals.append(np.nan)
        else:
            vals.append(cdf[idx])
    return pd.Series(vals, index=cum_triangle.index)


def compute_loss_ratio_ultimates(premiums: pd.Series, loss_ratio: float) -> pd.Series:
    return premiums * loss_ratio


def compute_bf(cum_triangle: pd.DataFrame, premiums: pd.Series, loss_ratio: float, factors_without_tail: List[float]) -> pd.DataFrame:
    latest = last_observed_by_row(cum_triangle)
    cdf_row = dev_cdf_by_row(cum_triangle, factors_without_tail)
    expected_ultimate = premiums.reindex(cum_triangle.index) * loss_ratio
    ultimate = latest + expected_ultimate * (1 - 1 / cdf_row)
    reserve = ultimate - latest
    return pd.DataFrame({
        "Latest": latest,
        "CDF": cdf_row,
        "Expected Ultimate (LR)": expected_ultimate,
        "Ultimate BF": ultimate,
        "Reserve BF": reserve,
    })


def compute_benktander(cum_triangle: pd.DataFrame, premiums: pd.Series, loss_ratio: float, factors_without_tail: List[float]) -> pd.DataFrame:
    cl_completed, latest, cl_ult, cl_reserve = complete_triangle_chain_ladder(cum_triangle, factors_without_tail)
    bf = compute_bf(cum_triangle, premiums, loss_ratio, factors_without_tail)
    cdf_row = bf["CDF"]
    ultimate = cl_ult * (1 / cdf_row) + bf["Ultimate BF"] * (1 - 1 / cdf_row)
    reserve = ultimate - latest
    return pd.DataFrame({
        "Latest": latest,
        "CDF": cdf_row,
        "Ultimate CL": cl_ult,
        "Ultimate BF": bf["Ultimate BF"],
        "Ultimate Benktander": ultimate,
        "Reserve Benktander": reserve,
    })


def compute_mack(cum_triangle: pd.DataFrame, factors_without_tail: List[float]) -> pd.DataFrame:
    tri = cum_triangle.copy().astype(float)
    n = tri.shape[0]
    m = tri.shape[1]
    # sigma_j^2 estimates
    sigma2 = np.full(m - 1, np.nan)
    lambdas = np.array(factors_without_tail, dtype=float)
    for j in range(m - 1):
        x = tri.iloc[:, j]
        y = tri.iloc[:, j + 1]
        valid = x.notna() & y.notna() & (x > 0)
        if valid.sum() <= 1:
            sigma2[j] = np.nan
            continue
        resid = y[valid] / x[valid] - lambdas[j]
        denom = valid.sum() - 1
        sigma2[j] = float((x[valid] * resid**2).sum() / denom)

    completed, latest, ultimate, reserve = complete_triangle_chain_ladder(tri, factors_without_tail)
    mseps = []
    stds = []
    cv = []
    for i in range(n):
        row = tri.iloc[i].to_numpy(dtype=float)
        valid = np.where(~np.isnan(row))[0]
        if len(valid) == 0:
            mseps.append(np.nan)
            stds.append(np.nan)
            cv.append(np.nan)
            continue
        last = valid[-1]
        c_last = row[last]
        if last >= m - 1:
            mseps.append(0.0)
            stds.append(0.0)
            cv.append(0.0)
            continue
        s = 0.0
        prod = 1.0
        for j in range(last, m - 1):
            prod *= lambdas[j]
            if np.isnan(sigma2[j]) or lambdas[j] == 0 or c_last <= 0:
                continue
            s += sigma2[j] / (lambdas[j] ** 2 * c_last)
        msep = (ultimate.iloc[i] ** 2) * s
        mseps.append(msep)
        stds.append(math.sqrt(max(msep, 0.0)))
        cv.append(0.0 if ultimate.iloc[i] == 0 else math.sqrt(max(msep, 0.0)) / ultimate.iloc[i])

    z95 = 1.96
    df = pd.DataFrame({
        "Latest": latest,
        "Ultimate Mack": ultimate,
        "Reserve Mack": reserve,
        "MSEP": mseps,
        "Std Dev": stds,
        "CV": cv,
    }, index=tri.index)
    df["IC95 Bas"] = df["Reserve Mack"] - z95 * df["Std Dev"]
    df["IC95 Haut"] = df["Reserve Mack"] + z95 * df["Std Dev"]
    return df


def bootstrap_chain_ladder(cum_triangle: pd.DataFrame, factors_without_tail: List[float], n_sims: int = 2000, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tri = cum_triangle.copy().astype(float)
    inc = to_incremental(tri)
    completed_cl, latest, cl_ult, cl_res = complete_triangle_chain_ladder(tri, factors_without_tail)
    fitted_inc = to_incremental(completed_cl)

    obs_mask = ~inc.isna()
    lower_mask = triangle_mask_lower(tri)

    fitted_obs = fitted_inc.where(obs_mask)
    resid = (inc - fitted_obs) / np.sqrt(np.abs(fitted_obs).replace(0, np.nan))
    resid_vals = resid.stack().dropna().to_numpy(dtype=float)
    if len(resid_vals) == 0:
        raise ValueError("Impossible de calculer le bootstrap : résidus indisponibles.")

    sims = []
    row_names = list(tri.index)
    for _ in range(n_sims):
        pseudo_inc = fitted_obs.copy()
        sampled = rng.choice(resid_vals, size=int(obs_mask.sum().sum()), replace=True)
        k = 0
        for i in range(pseudo_inc.shape[0]):
            for j in range(pseudo_inc.shape[1]):
                if obs_mask.iloc[i, j]:
                    base = pseudo_inc.iloc[i, j]
                    pseudo_inc.iloc[i, j] = max(base + sampled[k] * math.sqrt(abs(base)), 0.0)
                    k += 1
        pseudo_cum = to_cumulative(pseudo_inc)
        sim_factors = selected_factors_from_ratios(pseudo_cum, mode="pondérée", tail_factor=1.0).iloc[:-1].to_list()
        pseudo_completed, _, sim_ultimate, sim_reserve = complete_triangle_chain_ladder(pseudo_cum, sim_factors)
        sims.append(sim_reserve.to_numpy(dtype=float))

    sims_arr = np.vstack(sims)
    summary = pd.DataFrame(index=row_names)
    summary["Latest"] = latest
    summary["Reserve CL"] = cl_res
    summary["Bootstrap Mean"] = sims_arr.mean(axis=0)
    summary["Bootstrap Std"] = sims_arr.std(axis=0, ddof=1)
    summary["Bootstrap Median"] = np.median(sims_arr, axis=0)
    summary["Bootstrap Q75"] = np.quantile(sims_arr, 0.75, axis=0)
    summary["Bootstrap Q90"] = np.quantile(sims_arr, 0.90, axis=0)
    summary["Bootstrap Q95"] = np.quantile(sims_arr, 0.95, axis=0)

    dist = pd.DataFrame(sims_arr, columns=row_names)
    dist["Total Reserve"] = dist.sum(axis=1)
    return summary, dist


def linear_regression_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return a, b, r2


def make_excel_report(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31])
    return output.getvalue()


def read_uploaded_triangle(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xlsm", ".xls")):
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("Feuille Excel à utiliser", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet)
    else:
        raise ValueError("Format non supporté. Utilisez CSV ou Excel.")
    return sanitize_triangle(df)


# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    input_mode = st.radio(
        "Source du triangle",
        ["Coller les données", "Téléverser un fichier"],
        index=0,
    )
    data_type = st.selectbox("Type de triangle fourni", ["Cumulé", "Incrémental"])
    factor_mode = st.selectbox(
        "Sélection des facteurs Chain-Ladder",
        ["pondérée", "simple", "min", "max"],
        index=0,
    )
    tail_factor = st.number_input("Facteur de queue (tail factor)", min_value=1.0, value=1.0, step=0.01)
    default_loss_ratio = st.number_input("Loss Ratio / S/P retenu", min_value=0.0, value=0.60, step=0.01, format="%.4f")
    n_sims = st.slider("Nombre de simulations Bootstrap", 200, 5000, 1500, 100)
    seed = st.number_input("Seed Bootstrap", min_value=0, value=42, step=1)
    st.markdown("---")
    st.caption("Application personnalisée pour " + APP_AUTHOR)

# -------------------------
# Hero
# -------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1 style="margin:0;">📘 Provisionnement Non-Vie </h1>
        <p style="font-size:1.05rem;margin:0.4rem 0 0.15rem 0;">
            Conçu par <b>{APP_AUTHOR}</b> — triangle de liquidation, projection, diagnostics et comparaison multi-méthodes.
        </p>
        <p style="margin:0;opacity:0.92;">
            Méthodes intégrées : Chain-Ladder, Loss Ratio, Bornhuetter-Ferguson, Benktander, Mack et Bootstrap.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Cette application suit la logique du guide de provisionnement non-vie de l’Institut des actuaires : méthodes déterministes, méthodes stochastiques, segmentation, validation des hypothèses et qualité des données."
)

with st.expander("🧭 Étapes  à suivre pour utiliser cette application", expanded=False):
    st.markdown(
        """
        1. Importer ou coller le triangle de liquidation.  
        2. Indiquer s'il est **cumulé** ou **incrémental**.  
        3. Contrôler la qualité de la donnée.  
        4. Calculer les **coefficients de passage** et les **CDF**.  
        5. Projeter le triangle avec **Chain-Ladder**.  
        6. Ajouter les **primes acquises** pour Loss Ratio, BF et Benktander.  
        7. Mesurer l'incertitude avec **Mack** et **Bootstrap**.  
        8. Comparer les réserves par méthode, par origine et au total.  
        9. Télécharger un rapport Excel complet.  
        """
    )

# -------------------------
# Input area
# -------------------------
triangle_df = None
if input_mode == "Coller les données":
    sample_text = """Origine\t0\t1\t2\t3\t4\n2019\t120\t180\t210\t225\t230\n2020\t135\t195\t228\t240\t\n2021\t150\t216\t248\t\t\n2022\t165\t231\t\t\t\n2023\t180\t\t\t\t\n"""
    text = st.text_area(
        "Collez ici votre triangle (1re colonne = année d'origine, colonnes suivantes = développements)",
        value=sample_text,
        height=220,
    )
    if text.strip():
        triangle_df = parse_triangle_from_text(text)
else:
    uploaded = st.file_uploader("Téléversez un triangle CSV/Excel", type=["csv", "xlsx", "xls", "xlsm"])
    if uploaded is not None:
        triangle_df = read_uploaded_triangle(uploaded)

if triangle_df is None:
    st.stop()

if data_type == "Incrémental":
    inc_input = triangle_df.copy()
    cum_triangle = to_cumulative(triangle_df)
else:
    cum_triangle = triangle_df.copy()
    inc_input = to_incremental(cum_triangle)

# Premiums optional
with st.expander("💶 Primes acquises pour Loss Ratio / BF / Benktander", expanded=True):
    st.write("Saisissez une prime acquise par année d'origine. Si vous ne renseignez rien, une prime par défaut = dernier cumul observé sera proposée.")
    default_premiums = last_observed_by_row(cum_triangle).rename("Prime acquise").reset_index()
    default_premiums.columns = ["Origine", "Prime acquise"]
    premiums_edit = st.data_editor(default_premiums, num_rows="fixed", use_container_width=True)
    premiums_series = pd.Series(pd.to_numeric(premiums_edit["Prime acquise"], errors="coerce").values, index=premiums_edit["Origine"].astype(str))

# -------------------------
# Core calculations
# -------------------------
link_ratios = compute_link_ratios(cum_triangle)
selected = selected_factors_from_ratios(cum_triangle, mode=factor_mode, tail_factor=tail_factor)
selected_no_tail = normalize_factors_for_triangle(selected.iloc[:-1].to_list(), cum_triangle.shape[1])
completed_cl, latest, ultimate_cl, reserve_cl = complete_triangle_chain_ladder(cum_triangle, selected_no_tail)
row_cdf = dev_cdf_by_row(cum_triangle, selected_no_tail, tail_factor=tail_factor)
global_cdf = cdf_from_factors(selected_no_tail, tail_factor=tail_factor)

premiums_series = premiums_series.reindex(cum_triangle.index.astype(str))
premiums_series.index = cum_triangle.index
lr_ult = compute_loss_ratio_ultimates(premiums_series, default_loss_ratio)
bf_df = compute_bf(cum_triangle, premiums_series, default_loss_ratio, selected_no_tail)
benk_df = compute_benktander(cum_triangle, premiums_series, default_loss_ratio, selected_no_tail)
mack_df = compute_mack(cum_triangle, selected_no_tail)
bootstrap_summary, bootstrap_dist = bootstrap_chain_ladder(cum_triangle, selected_no_tail, n_sims=n_sims, seed=int(seed))

comparison = pd.DataFrame(index=cum_triangle.index)
comparison["Latest"] = latest
comparison["CL"] = reserve_cl
comparison["Loss Ratio"] = lr_ult - latest
comparison["Bornhuetter-Ferguson"] = bf_df["Reserve BF"]
comparison["Benktander"] = benk_df["Reserve Benktander"]
comparison["Mack"] = mack_df["Reserve Mack"]
comparison["Bootstrap Mean"] = bootstrap_summary["Bootstrap Mean"]
comparison.loc["TOTAL"] = comparison.sum(numeric_only=True)

# -------------------------
# KPIs
# -------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Réserve CL totale", format_num(reserve_cl.sum()))
with col2:
    st.metric("Réserve BF totale", format_num(bf_df["Reserve BF"].sum()))
with col3:
    st.metric("Réserve Mack totale", format_num(mack_df["Reserve Mack"].sum()))
with col4:
    st.metric("Bootstrap Q95 total", format_num(float(np.quantile(bootstrap_dist["Total Reserve"], 0.95))))

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Données",
    "2. Chain-Ladder",
    "3. Loss Ratio / BF / Benktander",
    "4. Mack",
    "5. Bootstrap",
    "6. Diagnostics",
    "7. Export",
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Triangle fourni")
        st.dataframe(triangle_df.style.format(format_num), use_container_width=True)
        st.markdown("#### Triangle incrémental")
        st.dataframe(inc_input.style.format(format_num), use_container_width=True)
    with c2:
        st.markdown("#### Triangle cumulé")
        st.dataframe(cum_triangle.style.format(format_num), use_container_width=True)
        qc = pd.DataFrame({
            "Contrôle": [
                "Valeurs négatives",
                "Lignes vides",
                "Colonnes vides",
                "Monotonicité cumulée",
            ],
            "Résultat": [
                "Oui" if (cum_triangle < 0).stack().any() else "Non",
                int(cum_triangle.isna().all(axis=1).sum()),
                int(cum_triangle.isna().all(axis=0).sum()),
                "OK" if ((cum_triangle.ffill(axis=1).diff(axis=1).fillna(0) >= 0).stack().all()) else "À vérifier",
            ],
        })
        st.markdown("#### Contrôles qualité")
        st.dataframe(qc, use_container_width=True, hide_index=True)

with tab2:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("#### Coefficients individuels")
        st.dataframe(link_ratios.style.format(lambda x: "" if pd.isna(x) else f"{x:.4f}"), use_container_width=True)
        st.markdown("#### Facteurs retenus")
        factors_df = pd.DataFrame({
            "Facteur": selected.index,
            "Valeur": selected.values,
        })
        st.dataframe(factors_df.style.format({"Valeur": "{:.4f}"}), use_container_width=True, hide_index=True)
        cdf_df = pd.DataFrame({
            "Développement": list(cum_triangle.columns) + ["Ultime"],
            "CDF": list(global_cdf) + [1.0],
        })
        st.markdown("#### CDF à l'ultime")
        st.dataframe(cdf_df.style.format({"CDF": "{:.4f}"}), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Triangle Chain-Ladder complété")
        st.dataframe(completed_cl.style.format(format_num), use_container_width=True)
        reserve_df = pd.DataFrame({
            "Latest": latest,
            "Ultimate CL": ultimate_cl,
            "Reserve CL": reserve_cl,
            "CDF": row_cdf,
        })
        st.markdown("#### Réserves par année d'origine")
        st.dataframe(reserve_df.style.format(format_num), use_container_width=True)

with tab3:
    st.markdown("#### Comparaison des méthodes déterministes")
    det = pd.DataFrame(index=cum_triangle.index)
    det["Latest"] = latest
    det["Prime acquise"] = premiums_series
    det["Ultimate LR"] = lr_ult
    det["Reserve LR"] = lr_ult - latest
    det["Ultimate BF"] = bf_df["Ultimate BF"]
    det["Reserve BF"] = bf_df["Reserve BF"]
    det["Ultimate Benktander"] = benk_df["Ultimate Benktander"]
    det["Reserve Benktander"] = benk_df["Reserve Benktander"]
    det.loc["TOTAL"] = det.sum(numeric_only=True)
    st.dataframe(det.style.format(format_num), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    methods = ["Reserve LR", "Reserve BF", "Reserve Benktander"]
    totals = [det.loc[cum_triangle.index, m].sum() for m in methods]
    ax.bar(methods, totals)
    ax.set_title("Réserve totale par méthode déterministe")
    ax.set_ylabel("Montant")
    plt.xticks(rotation=10)
    st.pyplot(fig)

with tab4:
    st.markdown("#### Résultats Mack")
    mack_display = mack_df.copy()
    mack_display.loc["TOTAL"] = mack_display.sum(numeric_only=True)
    st.dataframe(mack_display.style.format(format_num), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(mack_df.index.astype(str), mack_df["Std Dev"])
    ax.set_title("Écart-type Mack par année d'origine")
    ax.set_ylabel("Std Dev")
    ax.set_xlabel("Origine")
    st.pyplot(fig)

with tab5:
    st.markdown("#### Synthèse Bootstrap")
    boot_disp = bootstrap_summary.copy()
    boot_disp.loc["TOTAL"] = boot_disp.sum(numeric_only=True)
    st.dataframe(boot_disp.style.format(format_num), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.hist(bootstrap_dist["Total Reserve"], bins=40)
    ax.set_title("Distribution Bootstrap de la réserve totale")
    ax.set_xlabel("Réserve totale")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

    q_df = pd.DataFrame({
        "Statistique": ["Moyenne", "Médiane", "Q75", "Q90", "Q95"],
        "Valeur": [
            bootstrap_dist["Total Reserve"].mean(),
            bootstrap_dist["Total Reserve"].median(),
            bootstrap_dist["Total Reserve"].quantile(0.75),
            bootstrap_dist["Total Reserve"].quantile(0.90),
            bootstrap_dist["Total Reserve"].quantile(0.95),
        ],
    })
    st.dataframe(q_df.style.format({"Valeur": format_num}), use_container_width=True, hide_index=True)

with tab6:
    st.markdown("#### Validation Chain-Ladder")
    cols = list(cum_triangle.columns)
    if len(cols) >= 2:
        dev_choice = st.selectbox("Couple de développements à analyser", list(range(len(cols) - 1)), format_func=lambda j: f"{cols[j]} -> {cols[j+1]}")
        x = cum_triangle.iloc[:, dev_choice]
        y = cum_triangle.iloc[:, dev_choice + 1]
        valid = x.notna() & y.notna()
        x_vals = x[valid].to_numpy(dtype=float)
        y_vals = y[valid].to_numpy(dtype=float)
        a, b, r2 = linear_regression_stats(x_vals, y_vals)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pente estimée", "" if np.isnan(a) else f"{a:.4f}")
            st.metric("R²", "" if np.isnan(r2) else f"{r2:.4f}")
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.scatter(x_vals, y_vals)
            if not np.isnan(a):
                xx = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(xx, a * xx + b)
            ax.set_title(f"Régression {cols[dev_choice]} -> {cols[dev_choice + 1]}")
            ax.set_xlabel(str(cols[dev_choice]))
            ax.set_ylabel(str(cols[dev_choice + 1]))
            st.pyplot(fig)
        with c2:
            if not np.isnan(a):
                resid = y_vals - (a * x_vals + b)
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                ax.axhline(0, linestyle="--")
                ax.scatter(np.arange(len(resid)), resid)
                ax.set_title("Résidus de régression")
                ax.set_xlabel("Observation")
                ax.set_ylabel("Résidu")
                st.pyplot(fig)

    st.markdown("#### Comparaison globale des réserves")
    st.dataframe(comparison.style.format(format_num), use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    comp_plot = comparison.drop(index="TOTAL")[["CL", "Loss Ratio", "Bornhuetter-Ferguson", "Benktander", "Mack", "Bootstrap Mean"]]
    for col in comp_plot.columns:
        ax.plot(comp_plot.index.astype(str), comp_plot[col], marker="o", label=col)
    ax.set_title("Comparaison des réserves par origine")
    ax.set_xlabel("Origine")
    ax.set_ylabel("Réserve")
    ax.legend()
    st.pyplot(fig)

with tab7:
    st.markdown("#### Export Excel complet")
    report_bytes = make_excel_report({
        "triangle_input": triangle_df,
        "triangle_incremental": inc_input,
        "triangle_cumulative": cum_triangle,
        "link_ratios": link_ratios,
        "chain_ladder_completed": completed_cl,
        "bf": bf_df,
        "benktander": benk_df,
        "mack": mack_df,
        "bootstrap_summary": bootstrap_summary,
        "comparison": comparison,
    })
    st.download_button(
        "📥 Télécharger le rapport Excel",
        data=report_bytes,
        file_name="rapport_provisionnement_non_vie.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.code("streamlit run provisionnement_app.py", language="bash")
    st.markdown("#### Recommandations métier")
    st.markdown(
        """
        - Vérifier la **segmentation en groupes de risques homogènes** avant projection.
        - Retraiter les **sinistres graves** ou événements exceptionnels si nécessaire.
        - Contrôler la **qualité des données** : valeurs négatives, diagonales incohérentes, ruptures de gestion.
        - Éviter le **cherry-picking** : justifier le montant final retenu parmi les méthodes testées.
        """
    )

st.caption(
    "Référentiel métier intégré dans l'interface : Chain-Ladder, Loss Ratio, Bornhuetter-Ferguson, Benktander, Mack, Bootstrap, segmentation et qualité des données issus du guide de provisionnement non-vie de l'Institut des actuaires (février 2023)."
)
