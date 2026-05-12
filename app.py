# =========================================================
# DATA CHALLENGE ENSAR
# BASE TVS CHIRURGIENS-DENTISTES
# VERSION ULTRA ROBUSTE
# =========================================================

import pandas as pd
import numpy as np
import os
import re
import unicodedata

from difflib import SequenceMatcher

# =========================================================
# DOSSIER
# =========================================================

DOSSIER = r"C:\Users\belfo\Downloads\Nos Bases"

# =========================================================
# FONCTIONS
# =========================================================

def clean_cols(df):

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def clean_code(x):

    if pd.isna(x):
        return np.nan

    x = str(x).replace(".0", "").strip()

    if x == "" or x.lower() == "nan":
        return np.nan

    return x.zfill(5)


def clean_text(x):

    if pd.isna(x):
        return ""

    x = str(x).upper().strip()

    x = unicodedata.normalize(
        "NFKD",
        x
    ).encode(
        "ASCII",
        "ignore"
    ).decode("utf-8")

    remplacements = {

        "SAINT ": "ST ",
        "SAINTE ": "STE ",

        "LE ": "",
        "LA ": "",
        "LES ": "",

        "L ": "",
        "D ": "",

        "-": " ",
        "'": " "

    }

    for a, b in remplacements.items():

        x = x.replace(a, b)

    x = re.sub(r"[^A-Z0-9 ]", " ", x)

    x = re.sub(r"\s+", " ", x)

    return x.strip()


def similarity(a, b):

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def insee_plm(cp):

    if pd.isna(cp):
        return np.nan

    cp = str(cp).zfill(5)

    if cp.startswith("750"):
        return "751" + cp[-2:]

    if cp.startswith("130"):
        return "132" + cp[-2:]

    if cp.startswith("6900"):
        return "6938" + cp[-1]

    return np.nan

# =========================================================
# IMPORT
# =========================================================

print("\nImport des bases...")

ps = pd.read_csv(
    os.path.join(
        DOSSIER,
        "liste-ps-20260316-023101.csv"
    ),
    sep=";",
    dtype=str,
    low_memory=False
)

cp = pd.read_csv(
    os.path.join(
        DOSSIER,
        "20230823-communes-departement-region.csv"
    ),
    sep=",",
    dtype=str
)

cedex = pd.read_excel(
    os.path.join(
        DOSSIER,
        "correspondances_cedex_insee.xlsx"
    ),
    header=1,
    dtype=str
)

tvs = pd.read_csv(
    os.path.join(
        DOSSIER,
        "correspondance-tvs-communes-2018.csv"
    ),
    sep=",",
    dtype=str
)

pop = pd.read_excel(
    os.path.join(
        DOSSIER,
        "base-pop-historiques-1876-2023.xlsx"
    ),
    sheet_name="pop_1876_2023",
    skiprows=5,
    dtype={"CODGEO": str}
)

# =========================================================
# NORMALISATION
# =========================================================

ps = clean_cols(ps)
cp = clean_cols(cp)
cedex = clean_cols(cedex)
tvs = clean_cols(tvs)
pop = clean_cols(pop)

# =========================================================
# DENTISTES
# =========================================================

ps = ps[
    [
        "ps_activite_nom",
        "ps_activite_prenom",
        "specialite_code",
        "specialite_libelle",
        "coordonnees_code_postal",
        "coordonnees_ville",
        "nature_exercice_libelle"
    ]
].copy()

ps = ps.rename(columns={

    "ps_activite_nom": "nom",
    "ps_activite_prenom": "prenom",

    "coordonnees_code_postal": "code_postal",
    "coordonnees_ville": "ville",

    "nature_exercice_libelle": "nature_exercice"

})

# =========================================================
# SPECIALITES
# =========================================================

specialites = ["19", "53", "54"]

ps = ps[
    ps["specialite_code"]
    .isin(specialites)
].copy()

# =========================================================
# LIBERAUX
# =========================================================

ps = ps[
    ps["nature_exercice"]
    .astype(str)
    .str.contains(
        "lib",
        case=False,
        na=False
    )
].copy()

# =========================================================
# CLEAN
# =========================================================

ps["nom"] = (
    ps["nom"]
    .astype(str)
    .str.upper()
    .str.strip()
)

ps["prenom"] = (
    ps["prenom"]
    .astype(str)
    .str.upper()
    .str.strip()
)

ps["code_postal"] = (
    ps["code_postal"]
    .apply(clean_code)
)

ps["ville_clean"] = (
    ps["ville"]
    .apply(clean_text)
)

# =========================================================
# CODES POSTAUX VALIDES
# =========================================================

ps = ps[
    ps["code_postal"]
    .str.match(
        r"^[0-9]{5}$",
        na=False
    )
].copy()

# =========================================================
# DEDOUBLONNAGE
# =========================================================

ps = ps.drop_duplicates(
    subset=[
        "nom",
        "prenom",
        "code_postal"
    ]
)

ps = (
    ps
    .sort_values("code_postal")
    .drop_duplicates(
        subset=[
            "nom",
            "prenom"
        ],
        keep="last"
    )
)

print("\nDentistes uniques :")
print(len(ps))

# =========================================================
# BASE GEO
# =========================================================

cp = cp.rename(columns={
    "code_commune_insee": "insee"
})

cp["code_postal"] = (
    cp["code_postal"]
    .apply(clean_code)
)

cp["insee"] = (
    cp["insee"]
    .apply(clean_code)
)

# =========================================================
# TABLE LONGUE GEO
# =========================================================

geo1 = cp[
    [
        "code_postal",
        "insee",
        "nom_commune"
    ]
].rename(
    columns={
        "nom_commune": "ville"
    }
)

geo2 = cp[
    [
        "code_postal",
        "insee",
        "libelle_acheminement"
    ]
].rename(
    columns={
        "libelle_acheminement": "ville"
    }
)

geo_long = pd.concat(
    [
        geo1,
        geo2
    ],
    ignore_index=True
)

geo_long["ville_clean"] = (
    geo_long["ville"]
    .apply(clean_text)
)

geo_long = geo_long.drop_duplicates()

# =========================================================
# TABLE TVS UNIQUE
# =========================================================

tvs["insee"] = (
    tvs["insee"]
    .apply(clean_code)
)

table_tvs = (

    tvs[
        [
            "insee",
            "tvs"
        ]
    ]

    .dropna()

    .drop_duplicates(
        subset=["insee"]
    )

)

# =========================================================
# MATCH INSEE
# =========================================================

def trouver_meilleur_insee(row):

    cp_user = row["code_postal"]
    ville_user = row["ville_clean"]

    # PLM
    plm = insee_plm(cp_user)

    if pd.notna(plm):
        return plm

    candidats = geo_long[
        geo_long["code_postal"] == cp_user
    ]

    if len(candidats) == 0:
        return np.nan

    candidats = candidats.copy()

    candidats["score"] = candidats[
        "ville_clean"
    ].apply(
        lambda x: similarity(
            ville_user,
            x
        )
    )

    meilleur = candidats.sort_values(
        "score",
        ascending=False
    ).iloc[0]

    if meilleur["score"] >= 0.85:
        return meilleur["insee"]

    # fallback CP unique

    nb_insee = candidats["insee"].nunique()

    if nb_insee == 1:
        return candidats["insee"].iloc[0]

    return np.nan

# =========================================================
# APPLICATION
# =========================================================

print("\nRecherche INSEE...")

ps["insee_final"] = ps.apply(
    trouver_meilleur_insee,
    axis=1
)

# =========================================================
# CEDEX
# =========================================================

cedex = cedex.rename(columns={
    "cedex": "code_postal",
    "insee": "insee_cedex"
})

cedex["code_postal"] = (
    cedex["code_postal"]
    .apply(clean_code)
)

cedex["insee_cedex"] = (
    cedex["insee_cedex"]
    .apply(clean_code)
)

cedex = cedex.drop_duplicates(
    subset=["code_postal"]
)

ps = ps.merge(
    cedex,
    on="code_postal",
    how="left"
)

mask = (
    ps["insee_final"].isna()
    &
    ps["insee_cedex"].notna()
)

ps.loc[
    mask,
    "insee_final"
] = ps.loc[
    mask,
    "insee_cedex"
]

# =========================================================
# TVS
# =========================================================

ps = ps.merge(

    table_tvs,

    left_on="insee_final",
    right_on="insee",

    how="left"

)

# =========================================================
# CONTROLES
# =========================================================

print("\nTaux INSEE")
print(
    round(
        ps["insee_final"]
        .notna()
        .mean() * 100,
        2
    ),
    "%"
)

print("\nTaux TVS")
print(
    round(
        ps["tvs"]
        .notna()
        .mean() * 100,
        2
    ),
    "%"
)

# =========================================================
# DENTISTES PAR TVS
# =========================================================

dentistes_tvs = (

    ps[
        ps["tvs"].notna()
    ]

    .groupby("tvs")

    .size()

    .reset_index(name="nb_dentistes")

)

print("\nSomme dentistes :")
print(
    dentistes_tvs["nb_dentistes"]
    .sum()
)

# =========================================================
# POPULATION
# =========================================================

pop = pop.rename(columns={

    "codgeo": "insee",
    "libgeo": "commune",
    "pmun2023": "population_2023"

})

pop = pop[
    [
        "insee",
        "commune",
        "population_2023"
    ]
].copy()

pop["insee"] = (
    pop["insee"]
    .apply(clean_code)
)

pop["population_2023"] = pd.to_numeric(
    pop["population_2023"],
    errors="coerce"
)

# =========================================================
# POP -> TVS
# =========================================================

pop = pop.merge(

    table_tvs,

    on="insee",

    how="left"

)

population_tvs = (

    pop[
        pop["tvs"].notna()
    ]

    .groupby("tvs")["population_2023"]

    .sum()

    .reset_index()

)

# =========================================================
# BASE FINALE
# =========================================================

base_finale = population_tvs.merge(

    dentistes_tvs,

    on="tvs",

    how="left"

)

base_finale["nb_dentistes"] = (
    base_finale["nb_dentistes"]
    .fillna(0)
)

# =========================================================
# KPI
# =========================================================

base_finale["densite_100k"] = np.where(

    base_finale["population_2023"] > 0,

    (
        base_finale["nb_dentistes"]
        / base_finale["population_2023"]
    ) * 100000,

    0

)

base_finale["population_par_dentiste"] = np.where(

    base_finale["nb_dentistes"] > 0,

    (
        base_finale["population_2023"]
        / base_finale["nb_dentistes"]
    ),

    base_finale["population_2023"]

)

base_finale["absence_dentiste"] = np.where(

    base_finale["nb_dentistes"] == 0,
    1,
    0

)

conditions = [

    base_finale["densite_100k"] == 0,

    base_finale["densite_100k"] < 30,

    base_finale["densite_100k"] < 50,

    base_finale["densite_100k"] < 80

]

valeurs = [

    "Désert médical sévère",
    "Sous-doté",
    "Intermédiaire",
    "Correctement doté"

]

base_finale["categorie_desert_medical"] = np.select(

    conditions,
    valeurs,
    default="Bien doté"

)

max_densite = (
    base_finale["densite_100k"]
    .max()
)

base_finale["score_accessibilite"] = np.where(

    max_densite > 0,

    (
        base_finale["densite_100k"]
        / max_densite
    ) * 100,

    0

)

# =========================================================
# ARRONDIS
# =========================================================

base_finale["densite_100k"] = (
    base_finale["densite_100k"]
    .round(2)
)

base_finale["population_par_dentiste"] = (
    base_finale["population_par_dentiste"]
    .round(0)
)

base_finale["score_accessibilite"] = (
    base_finale["score_accessibilite"]
    .round(2)
)

# =========================================================
# TRI
# =========================================================

base_finale = base_finale.sort_values(
    "densite_100k",
    ascending=False
)

# =========================================================
# TVS SUSPECTES
# =========================================================

suspectes = base_finale[
    (
        base_finale["population_2023"] > 20000
    )
    &
    (
        base_finale["nb_dentistes"] == 0
    )
]

print("\nTVS suspectes")
print(
    suspectes[
        [
            "tvs",
            "population_2023",
            "nb_dentistes"
        ]
    ].head(50)
)

# =========================================================
# EXPORT
# =========================================================

fichier_export = os.path.join(
    DOSSIER,
    "base_kpi_tvs_ultra_robuste.csv"
)

try:

    base_finale.to_csv(

        fichier_export,

        sep=";",
        index=False,
        encoding="utf-8-sig"

    )

except PermissionError:

    fichier_export = os.path.join(
        DOSSIER,
        "base_kpi_tvs_ultra_robuste_v2.csv"
    )

    base_finale.to_csv(

        fichier_export,

        sep=";",
        index=False,
        encoding="utf-8-sig"

    )

# =========================================================
# FIN
# =========================================================

print("\nExport terminé")
print(fichier_export)

print("\nDimensions")
print(base_finale.shape)

print("\nValeurs manquantes")
print(base_finale.isna().sum())
