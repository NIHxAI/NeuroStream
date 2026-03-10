import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    StandardScaler
)
from pycombat import Combat
from PIL.Image import open

path='./assets'

ind=[
    'C_ID','epid','session_id','R_ID','session','ck_dcode','AS_DATA_CLASS','AS_EDATE'
]

exotic=["cretn_tr1","cretn_tr2"]
grouper="cohort"

@st.cache_data
def read(deta_path=path) -> pd.DataFrame:
    deta_files = [q.path for q in os.scandir(deta_path) if q.name.endswith(".f")]
    if not deta_files:
        raise FileNotFoundError("No .f files found in assets directory.")
    df = pd.concat([pd.read_feather(p) for p in deta_files], axis=0)
    
    cols_to_drop = [c for c in ind if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df

@st.cache_data
def get_col(deta) -> tuple:
    deta_site = deta.loc[:, grouper].unique()
    
    deta_cov_col = [q for q in deta.columns if q.startswith('cov_') or q.lower() == 'age']
    deta_contigous_col = deta.select_dtypes(include=[float, int]).columns.to_list()
    
    deta_vol_col = [
        q for q in deta_contigous_col 
        if str(q)[-1].isnumeric() and q not in exotic and q not in deta_cov_col
    ]
    
    deta_categorical_col = [
        c for c in deta.columns 
        if c not in deta_vol_col and c.lower() != 'age' and c.lower() != 'cov_age'
    ]
    
    if grouper not in deta_categorical_col:
        deta_categorical_col.append(grouper)
        
    return deta_site, deta_categorical_col, deta_contigous_col, deta_vol_col

@st.cache_data
def get_var(deta=None) -> tuple:
    if deta is None:
        deta = read()
    deta.index = range(deta.shape[0])
    site, cat, cont, vol = get_col(deta)
    return deta, site, cat, cont, vol

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
    deta_vol_col: list,
    covariates: list = None
) -> pd.DataFrame:
    final = deta.copy()

    if how == "Log Transform (log1p)":
        final[deta_contigous_col] = safe_log1p(deta, deta_contigous_col, eps=1e-6, do_zscore=False)

    elif how == "Log Transform + Z-score":
        final[deta_contigous_col] = safe_log1p(deta, deta_contigous_col, eps=1e-6, do_zscore=True)

    elif how == "Scale (Z-score)":
        scaler = StandardScaler()
        X = deta.loc[:, deta_contigous_col].to_numpy(dtype=float)
        final[deta_contigous_col] = scaler.fit_transform(X)

    elif "Combat" in how:
        if not covariates:
            Xc = None
        else:
            cov_t_list = []
            for col in covariates:
                if deta[col].dtype in ['int64', 'int32'] or deta[col].nunique() < 10:
                    coder = OneHotEncoder(sparse_output=False, drop='first')
                    encoded = coder.fit_transform(deta[[col]])
                    cov_t_list.append(encoded)
                else:
                    pt = PowerTransformer()
                    scaled = pt.fit_transform(deta[[col]].astype(float))
                    cov_t_list.append(scaled)
            Xc = np.concatenate(cov_t_list, axis=1).astype("f4") if cov_t_list else None

        stabiliser = Combat()
        X = deta.loc[:, deta_vol_col].to_numpy(dtype=float)
        Xb = deta.loc[:, grouper].to_numpy(dtype="U16")
        
        Xt = PowerTransformer().fit_transform(X)
        Xts = stabiliser.fit_transform(Xt, Xb, None, Xc)

        final[deta_vol_col] = Xts

    elif how == "divided by intracranial volume":
        icv_cols = [c for c in deta.columns if "icv" in c.lower()]
        if not icv_cols:
            raise ValueError("ICV column not found")
        icv_col = icv_cols[0]
        vol_no_icv = [c for c in deta_vol_col if c != icv_col]
        final[vol_no_icv] = deta.loc[:, vol_no_icv].div(deta[icv_col], axis=0)

    return final

@st.cache_data
def get_noe_image(deta_path=path)->dict:
    return {q.name.replace('.png',''):open(q.path) for q in os.scandir(deta_path) if q.name.endswith('.png')}

@st.cache_data
def trim(deta, deta_vol_col, gizun=.001) -> pd.DataFrame:
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
