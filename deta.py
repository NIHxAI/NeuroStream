import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    StandardScaler
)
from pycombat import Combat # https://github.com/CoAxLab/pycombat
from PIL.Image import open

path='./assets'

# Unusable Columns
ind=[
    'C_ID','epid','session_id','R_ID','session','ck_dcode','AS_DATA_CLASS','AS_EDATE'
]

# Exotic Columns
exotic=["cretn_tr1","cretn_tr2"]

# Grouper Column
grouper="cohort"

@st.cache_data
def read(deta_path=path) -> pd.DataFrame:
    deta_files = [q.path for q in os.scandir(deta_path) if q.name.endswith(".f")]

    if not deta_files:
        raise FileNotFoundError(
            "No .f files found in assets directory. "
            "Please upload a dataset in the 'Upload Dataset' tab."
        )

    df = pd.concat(
        [pd.read_feather(p) for p in deta_files],
        axis=0
    )

    cols_to_drop = [c for c in ind if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df

@st.cache_data
def get_col(deta)->tuple:
    deta_site=deta.loc[:, grouper].unique()
    deta_categorical_col=[*deta.select_dtypes(int).columns.to_list(), grouper]
    deta_contigous_col=deta.select_dtypes(float).columns.to_list()
    deta_vol_col=[q for q in deta_contigous_col if q[-1].isnumeric() and q not in exotic]
    
    return (
        deta_site,
        deta_categorical_col,
        deta_contigous_col,
        deta_vol_col
    )

@st.cache_data
def get_var(deta=None)->tuple:
    
    if deta is None:
        deta=read()
    
    deta.index=range(deta.shape[0])
    
    (
        deta_site,
        deta_categorical_col,
        deta_contigous_col,
        deta_vol_col
    )=get_col(deta)
    
    return (
        deta,
        deta_site,
        deta_categorical_col,
        deta_contigous_col,
        deta_vol_col
    )

@st.cache_data
def safe_log1p(df: pd.DataFrame, cols, eps=1e-6, do_zscore=False):
    X = df.loc[:, cols].to_numpy(dtype=float)
    mins = X.min(axis=0)
    shift = np.where(mins <= 0, -mins + eps, 0.0)
    X_log = np.log1p(X + shift)
    if do_zscore:
        X_log = StandardScaler().fit_transform(X_log)
    return pd.DataFrame(X_log, columns=cols)

@st.cache_data
def transform(
    deta: pd.DataFrame,
    how: str,
    deta_contigous_col: list,
    deta_vol_col: list
) -> pd.DataFrame:
    meta_df = pd.concat([deta.loc[:, grouper], deta.select_dtypes("int")], axis=1)

    # ----------------------------------------------------------------------
    # 1) Log only / Log + Z-score / Scale only
    # ----------------------------------------------------------------------
    if how == "Log Transform (log1p)":
        transformed = safe_log1p(deta, deta_contigous_col, eps=1e-6, do_zscore=False)
        final = pd.concat([meta_df, transformed], axis=1)

    elif how == "Log Transform + Z-score":
        transformed = safe_log1p(deta, deta_contigous_col, eps=1e-6, do_zscore=True)
        final = pd.concat([meta_df, transformed], axis=1)

    elif how == "Scale (Z-score)":
        scaler = StandardScaler()
        X = deta.loc[:, deta_contigous_col].to_numpy(dtype=float)
        Xz = scaler.fit_transform(X)
        transformed = pd.DataFrame(Xz, columns=deta_contigous_col, index=deta.index)
        final = pd.concat([meta_df, transformed], axis=1)

    # ----------------------------------------------------------------------
    # 2) Batch correction (ComBat with covariates: age, sex)
    # ----------------------------------------------------------------------
    elif how == "Combat (covariate: gender, age)":
        coder = OneHotEncoder(sparse_output=False)
        pt_feat = PowerTransformer()  
        stabiliser = Combat()

        X = deta.loc[:, deta_vol_col].to_numpy(dtype=float)
        Xv = np.asarray(deta_vol_col, dtype="U32")

        Xb = deta.loc[:, grouper].to_numpy(dtype="U16")

        Xc_sex = coder.fit_transform(deta["gender"].to_numpy().reshape(-1, 1))[:, [0]]

        batches = (
            deta[grouper]
            .astype(str)
            .dropna()
            .unique()
            .tolist()
        )

        Xc_age = np.full((deta.shape[0], 1), np.nan, dtype=float)

        for b in batches:
            mask = (deta[grouper].astype(str) == b).to_numpy()
            age_b = deta.loc[mask, "age"].to_numpy(dtype=float).reshape(-1, 1)

            if age_b.shape[0] < 3:
                Xc_age[mask, 0] = age_b[:, 0]
                continue

            pt_age = PowerTransformer()
            Xc_age_b = pt_age.fit_transform(age_b)
            Xc_age[mask, 0] = Xc_age_b[:, 0]

        Xc = np.concatenate([Xc_sex, Xc_age], axis=1).astype("f4")
        Xt = pt_feat.fit_transform(X)
        Xts = stabiliser.fit_transform(Xt, Xb, None, Xc)

        transformed = pd.DataFrame(Xts, columns=Xv, index=deta.index)
        final = pd.concat([meta_df, transformed], axis=1)

    # ----------------------------------------------------------------------
    # 3) Divide by intracranial volume 
    # ----------------------------------------------------------------------
    elif how == "divided by intracranial volume":
        icv_cols = [c for c in deta.columns if "icv" in c.lower()]
        if not icv_cols:
            raise ValueError("ICV column not found (expects substring 'icv' in column name).")
        icv_col = icv_cols[0]

        vol_no_icv = [c for c in deta_vol_col if c != icv_col]

        divided = deta.loc[:, vol_no_icv].div(deta[icv_col], axis=0)

        other_cont_cols = [c for c in deta_contigous_col if c not in vol_no_icv and c != icv_col]
        keep_cont = deta.loc[:, other_cont_cols] if other_cont_cols else pd.DataFrame(index=deta.index)

        final = pd.concat([meta_df, keep_cont, divided], axis=1)

    else:
        final = deta.copy()

    return final

@st.cache_data
def get_noe_image(deta_path=path)->dict:
    return {q.name.replace('.png',''):open(q.path) for q in os.scandir(deta_path) if q.name.endswith('.png')}

@st.cache_data
def trim(
    deta,
    deta_vol_col,
    gizun=.001
)->pd.DataFrame:
    '''Took 569 ± 23.9 ms '''

    deta_vol_col=[q for q in deta.columns if q in deta_vol_col]
    
    _lim=lambda q:(q.quantile(gizun), q.quantile(1-gizun))
    lim=deta[deta_vol_col].apply(_lim).to_dict("list")
    
    for col in lim.keys():
        _arr=deta.loc[:,col].to_numpy(dtype=np.float32)
        
        _arrLower=_arr<lim[col][0]
        _arrUpper=_arr>lim[col][1]
        
        _arr[_arrLower]=np.nan
        _arr[_arrUpper]=np.nan
        
        deta.loc[:,col]=_arr
    
    return deta
