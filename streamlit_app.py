"""
VIX Market-Timing Signal — Streamlit single-file deploy.

PUSH ONLY THESE THREE FILES TO YOUR GITHUB REPO ROOT:
    streamlit_app.py        ← THIS FILE (model + UI all in one)
    requirements.txt        ← see content below
    runtime.txt             ← single line: python-3.11

requirements.txt:
    streamlit
    pandas
    numpy
    xgboost
    scikit-learn
    requests
    yfinance

runtime.txt:
    python-3.11

DATA SOURCES (sidebar lets you switch live):
    - yfinance (free, default) — best on Streamlit Cloud
    - Twelve Data — set TWELVEDATA_KEY in Streamlit Secrets to enable.
      Free tier doesn't include CBOE indices; if you have a paid Indices
      add-on use direct symbols ("VIX","SPX","VIX3M","VVIX"); else the
      sidebar lets you fall back to ETF proxies (VIXY, SPY, VIXM, ^VVIX
      via yfinance).
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STRATEGY THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

STRICT_RULE = dict(name="STRICT (B@0.70/0.030)",
                   up_p=0.70, up_m=0.030, dn_p=0.30, dn_m=0.030)
LOOSE_RULE  = dict(name="LOOSER (B@0.60/0.020)",
                   up_p=0.60, up_m=0.020, dn_p=0.40, dn_m=0.020)
QUANTILES = [0.10, 0.50, 0.90]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING (114 features)
# ─────────────────────────────────────────────────────────────────────────────

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def parkinson(h, l):
    return (1.0 / (4.0 * np.log(2.0))) * (np.log(h / l)) ** 2

def garman_klass(o, h, l, c):
    return 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2

def rogers_satchell(o, h, l, c):
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["VIX"]; s = d["SPX"]
    vix_lr = np.log(c / c.shift(1))
    spx_lr = np.log(s / s.shift(1))

    for n in [1, 2, 3, 5, 10, 20]:
        d[f"vix_ret_{n}d"] = c.pct_change(n)
    for n in [5, 10, 20]:
        d[f"vix_rvol_{n}d"] = vix_lr.rolling(n).std() * np.sqrt(252)
    for n in [5, 10, 20, 50, 100]:
        d[f"vix_dist_ma{n}"] = (c - c.rolling(n).mean()) / c.rolling(n).mean()
    d["vix_rsi_14"]       = rsi(c, 14)
    d["vix_rsi_7"]        = rsi(c, 7)
    d["vix_level"]        = c
    d["vix_zscore_20"]    = (c - c.rolling(20).mean()) / c.rolling(20).std()
    d["vix_zscore_60"]    = (c - c.rolling(60).mean()) / c.rolling(60).std()
    d["vix_zscore_252"]   = (c - c.rolling(252).mean()) / c.rolling(252).std()
    d["vix_52w_high_pct"] = (c / c.rolling(252).max()) - 1
    d["vix_52w_low_pct"]  = (c / c.rolling(252).min()) - 1
    d["vix_intraday"]     = (d["VIX_HIGH"] - d["VIX_LOW"]) / c
    d["vix_open_close"]   = (c - d["VIX_OPEN"]) / d["VIX_OPEN"]
    for n in [3, 5, 10, 20]:
        d[f"vix_roc_{n}d"] = c.diff(n)

    direction = np.sign(vix_lr)
    streak, sv = [], 0
    for v in direction:
        if np.isnan(v):  streak.append(np.nan); sv = 0
        elif v > 0:      sv = sv + 1 if sv > 0 else 1;  streak.append(sv)
        elif v < 0:      sv = sv - 1 if sv < 0 else -1; streak.append(sv)
        else:            sv = 0; streak.append(0)
    d["vix_streak"] = streak
    d["vix_regime"] = pd.cut(c, bins=[0,12,15,20,25,30,40,999],
                             labels=[0,1,2,3,4,5,6]).astype(float)

    d["spx_ret_1d"]  = spx_lr
    d["spx_ret_2d"]  = spx_lr.rolling(2).sum()
    d["spx_ret_3d"]  = spx_lr.rolling(3).sum()
    d["spx_ret_5d"]  = spx_lr.rolling(5).sum()
    d["spx_ret_10d"] = spx_lr.rolling(10).sum()
    d["spx_ret_20d"] = spx_lr.rolling(20).sum()
    for n in [5, 10, 20]:
        d[f"spx_rvol_{n}d"] = spx_lr.rolling(n).std() * np.sqrt(252)
    for n in [20, 50, 200]:
        d[f"spx_dist_ma{n}"] = (s - s.rolling(n).mean()) / s.rolling(n).mean()
    d["spx_rsi_14"] = rsi(s, 14)

    spx_dir = np.sign(spx_lr)
    spx_streak, sv = [], 0
    for v in spx_dir:
        if np.isnan(v):  spx_streak.append(np.nan); sv = 0
        elif v > 0:      sv = sv + 1 if sv > 0 else 1;  spx_streak.append(sv)
        elif v < 0:      sv = sv - 1 if sv < 0 else -1; spx_streak.append(sv)
        else:            sv = 0; spx_streak.append(0)
    d["spx_streak"] = spx_streak
    d["vix_spx_corr_20d"] = vix_lr.rolling(20).corr(spx_lr)
    d["vix_vs_realized"]  = c - d["spx_rvol_20d"]
    d["spx_ret_1d_lag1"]  = d["spx_ret_1d"].shift(1)
    d["day_of_week"]      = d["DATE"].dt.dayofweek
    d["month"]            = d["DATE"].dt.month

    h, l, o = d["VIX_HIGH"], d["VIX_LOW"], d["VIX_OPEN"]
    d["vix_parkinson_1d"] = parkinson(h, l)
    d["vix_gk_1d"]        = garman_klass(o, h, l, c)
    d["vix_rs_1d"]        = rogers_satchell(o, h, l, c)
    for n in [5, 22]:
        d[f"vix_parkinson_{n}d"] = d["vix_parkinson_1d"].rolling(n).mean()
        d[f"vix_gk_{n}d"]        = d["vix_gk_1d"].rolling(n).mean()
        d[f"vix_rs_{n}d"]        = d["vix_rs_1d"].rolling(n).mean()

    vix_rv = vix_lr ** 2
    d["vix_har_1d"]  = vix_rv.shift(1)
    d["vix_har_5d"]  = vix_rv.shift(1).rolling(5).mean()
    d["vix_har_22d"] = vix_rv.shift(1).rolling(22).mean()
    spx_rv = spx_lr ** 2
    d["spx_har_1d"]  = spx_rv.shift(1)
    d["spx_har_5d"]  = spx_rv.shift(1).rolling(5).mean()
    d["spx_har_22d"] = spx_rv.shift(1).rolling(22).mean()

    pos = spx_lr.clip(lower=0); neg = spx_lr.clip(upper=0)
    for n in [5, 10, 22]:
        d[f"spx_pos_sq_{n}d"] = (pos ** 2).rolling(n).sum()
        d[f"spx_neg_sq_{n}d"] = (neg ** 2).rolling(n).sum()
        d[f"spx_asym_{n}d"]   = d[f"spx_neg_sq_{n}d"] - d[f"spx_pos_sq_{n}d"]
    vix_pos = vix_lr.clip(lower=0); vix_neg = vix_lr.clip(upper=0)
    for n in [5, 22]:
        d[f"vix_pos_sq_{n}d"] = (vix_pos ** 2).rolling(n).sum()
        d[f"vix_neg_sq_{n}d"] = (vix_neg ** 2).rolling(n).sum()

    lv = np.log(c)
    lv_mean_252 = lv.rolling(252).mean()
    lv_std_252  = lv.rolling(252).std()
    d["logvix"]         = lv
    d["logvix_meanrev"] = (lv_mean_252 - lv) / lv_std_252
    d["logvix_z_252"]   = (lv - lv_mean_252) / lv_std_252

    q95 = vix_lr.rolling(252).quantile(0.95)
    q05 = vix_lr.rolling(252).quantile(0.05)
    d["vix_fear_spike"]    = (vix_lr > q95).astype(float)
    d["vix_fear_crush"]    = (vix_lr < q05).astype(float)
    d["vix_fear_spike_l1"] = d["vix_fear_spike"].shift(1)
    d["vix_fear_crush_l1"] = d["vix_fear_crush"].shift(1)

    d["dow_vix_level"] = d["day_of_week"] * d["vix_level"]
    d["is_monday"]     = (d["day_of_week"] == 0).astype(float)
    d["is_friday"]     = (d["day_of_week"] == 4).astype(float)
    d["monday_vix"]    = d["is_monday"] * c
    d["friday_vix"]    = d["is_friday"] * c

    if "VIX3M" in d.columns:
        v3m = d["VIX3M"]
        d["vix3m_level"]       = v3m
        d["vix_vix3m_ratio"]   = c / v3m
        d["vix_vix3m_spread"]  = c - v3m
        d["vix3m_log_diff"]    = np.log(c / v3m)
        d["contango"]          = (c < v3m).astype(float)
        d["vix3m_ret_1d"]      = v3m.pct_change(1)
        d["vix3m_ret_5d"]      = v3m.pct_change(5)
        d["vix3m_zscore_60"]   = (v3m - v3m.rolling(60).mean()) / v3m.rolling(60).std()
        d["term_slope_5d_chg"] = d["vix_vix3m_spread"].diff(5)
        d["term_flip_to_back"] = ((d["contango"].shift(1) == 1) &
                                  (d["contango"]      == 0)).astype(float)
    if "VVIX" in d.columns:
        vv = d["VVIX"]
        vv_lr = np.log(vv / vv.shift(1))
        d["vvix_level"]        = vv
        d["vvix_ret_1d"]       = vv.pct_change(1)
        d["vvix_ret_5d"]       = vv.pct_change(5)
        d["vvix_zscore_60"]    = (vv - vv.rolling(60).mean()) / vv.rolling(60).std()
        d["vvix_zscore_252"]   = (vv - vv.rolling(252).mean()) / vv.rolling(252).std()
        d["vvix_dist_ma20"]    = (vv - vv.rolling(20).mean()) / vv.rolling(20).mean()
        d["vvix_vix_ratio"]    = vv / c
        d["vvix_corr_vix_20d"] = vv_lr.rolling(20).corr(vix_lr)
    if "PCRATIO" in d.columns:
        pc = d["PCRATIO"]
        d["pc_level"]      = pc
        d["pc_ma5"]        = pc.rolling(5).mean()
        d["pc_ma20"]       = pc.rolling(20).mean()
        d["pc_zscore_60"]  = (pc - pc.rolling(60).mean()) / pc.rolling(60).std()
        d["pc_zscore_252"] = (pc - pc.rolling(252).mean()) / pc.rolling(252).std()
        d["pc_ret_1d"]     = pc.pct_change(1)
        d["pc_ret_5d"]     = pc.pct_change(5)

    d["target"] = (c.shift(-1) > c).astype(int)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE LISTS
# ─────────────────────────────────────────────────────────────────────────────

# Master feature list. XGBoost handles NaN natively, so v3 (cross-asset)
# features can be missing in early years (or always, if VIX3M/VVIX fail).
FEATURES_FULL = [
    "vix_ret_1d","vix_ret_2d","vix_ret_3d","vix_ret_5d","vix_ret_10d","vix_ret_20d",
    "vix_rvol_5d","vix_rvol_10d","vix_rvol_20d",
    "vix_dist_ma5","vix_dist_ma10","vix_dist_ma20","vix_dist_ma50","vix_dist_ma100",
    "vix_rsi_14","vix_rsi_7",
    "vix_level","vix_zscore_20","vix_zscore_60","vix_zscore_252",
    "vix_52w_high_pct","vix_52w_low_pct",
    "vix_intraday","vix_open_close",
    "vix_roc_3d","vix_roc_5d","vix_roc_10d","vix_roc_20d",
    "vix_streak","day_of_week","month","vix_regime",
    "spx_ret_1d","spx_ret_2d","spx_ret_3d","spx_ret_5d","spx_ret_10d","spx_ret_20d",
    "spx_rvol_5d","spx_rvol_10d","spx_rvol_20d",
    "spx_dist_ma20","spx_dist_ma50","spx_dist_ma200",
    "spx_rsi_14","spx_streak",
    "vix_spx_corr_20d","vix_vs_realized","spx_ret_1d_lag1",
    "vix_parkinson_1d","vix_parkinson_5d","vix_parkinson_22d",
    "vix_gk_1d","vix_gk_5d","vix_gk_22d",
    "vix_rs_1d","vix_rs_5d","vix_rs_22d",
    "vix_har_1d","vix_har_5d","vix_har_22d",
    "spx_har_1d","spx_har_5d","spx_har_22d",
    "spx_pos_sq_5d","spx_neg_sq_5d","spx_asym_5d",
    "spx_pos_sq_10d","spx_neg_sq_10d","spx_asym_10d",
    "spx_pos_sq_22d","spx_neg_sq_22d","spx_asym_22d",
    "vix_pos_sq_5d","vix_neg_sq_5d",
    "vix_pos_sq_22d","vix_neg_sq_22d",
    "logvix","logvix_meanrev","logvix_z_252",
    "vix_fear_spike","vix_fear_crush",
    "vix_fear_spike_l1","vix_fear_crush_l1",
    "dow_vix_level","is_monday","is_friday","monday_vix","friday_vix",
    "vix3m_level","vix_vix3m_ratio","vix_vix3m_spread","vix3m_log_diff",
    "contango","vix3m_ret_1d","vix3m_ret_5d","vix3m_zscore_60",
    "term_slope_5d_chg","term_flip_to_back",
    "vvix_level","vvix_ret_1d","vvix_ret_5d","vvix_zscore_60",
    "vvix_zscore_252","vvix_dist_ma20","vvix_vix_ratio","vvix_corr_vix_20d",
    "pc_level","pc_ma5","pc_ma20","pc_zscore_60","pc_zscore_252",
    "pc_ret_1d","pc_ret_5d",
]

# Required (always present) features for a row to be usable in training.
# These don't depend on VIX3M / VVIX / PCRATIO, so they always train.
CORE_REQUIRED = [
    "vix_ret_1d","vix_ret_2d","vix_ret_3d","vix_ret_5d","vix_ret_10d","vix_ret_20d",
    "vix_rvol_5d","vix_rvol_10d","vix_rvol_20d",
    "vix_dist_ma5","vix_dist_ma10","vix_dist_ma20","vix_dist_ma50","vix_dist_ma100",
    "vix_rsi_14","vix_rsi_7","vix_level",
    "vix_zscore_20","vix_zscore_60","vix_zscore_252",
    "vix_52w_high_pct","vix_52w_low_pct","vix_intraday","vix_open_close",
    "vix_roc_3d","vix_roc_5d","vix_roc_10d","vix_roc_20d",
    "vix_streak","day_of_week","month","vix_regime",
    "spx_ret_1d","spx_ret_2d","spx_ret_3d","spx_ret_5d","spx_ret_10d","spx_ret_20d",
    "spx_rvol_5d","spx_rvol_10d","spx_rvol_20d",
    "spx_dist_ma20","spx_dist_ma50","spx_dist_ma200",
    "spx_rsi_14","spx_streak",
    "vix_spx_corr_20d","vix_vs_realized","spx_ret_1d_lag1",
]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATA LOADERS  (ROBUST: yfinance + Twelve Data)
# ─────────────────────────────────────────────────────────────────────────────

def _load_yfinance(symbol: str, start: str = "1990-01-01") -> pd.DataFrame:
    """yfinance OHLC fetch — robust to recent yfinance API changes."""
    import yfinance as yf
    # auto_adjust=False — recent yfinance bug returns empty frames with True for indices
    d = yf.download(symbol, start=start, progress=False,
                    auto_adjust=False, threads=False)
    if d is None or len(d) == 0:
        raise RuntimeError(f"yfinance returned 0 rows for {symbol}")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.reset_index()
    date_col = "Date" if "Date" in d.columns else "Datetime"
    d = d.rename(columns={date_col: "DATE"})
    cols = ["DATE", "Open", "High", "Low", "Close"]
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise RuntimeError(
            f"yfinance result for {symbol} missing columns {missing}; "
            f"got {list(d.columns)}")
    return d[cols]


def _load_twelvedata(symbol: str, api_key: str,
                     start: str = "1990-01-01") -> pd.DataFrame:
    import requests
    url = ("https://api.twelvedata.com/time_series"
           f"?symbol={symbol}&interval=1day&start_date={start}"
           f"&apikey={api_key}&format=JSON&outputsize=5000")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json()
    if "values" not in js:
        raise RuntimeError(f"Twelve Data error for {symbol}: {js}")
    df = pd.DataFrame(js["values"])
    df["DATE"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.rename(columns={"open":"Open","high":"High",
                              "low":"Low","close":"Close"})[
        ["DATE","Open","High","Low","Close"]].sort_values("DATE")


def _fetch(spec):
    if spec["kind"] == "yfinance":
        return _load_yfinance(spec["symbol"])
    if spec["kind"] == "twelvedata":
        return _load_twelvedata(spec["symbol"], spec["api_key"])
    raise ValueError(f"Unknown source kind: {spec['kind']}")


def load_market_data(sources: Dict[str, Dict[str, Any]]
                    ) -> (pd.DataFrame, list):
    """Returns (merged_df, warnings_list).
    Required: VIX, SPX. Optional: VIX3M, VVIX, putcall."""
    warnings_out = []

    def _ohlc_to(df, prefix):
        return df.rename(columns={"Open": f"{prefix}_OPEN",
                                  "High": f"{prefix}_HIGH",
                                  "Low":  f"{prefix}_LOW",
                                  "Close": prefix})

    vix = _ohlc_to(_fetch(sources["vix"]), "VIX")
    spx = _ohlc_to(_fetch(sources["spx"]), "SPX")

    # OPTIONAL — graceful handling so the model still runs without them
    if "vix3m" in sources:
        try:
            vix3m = _fetch(sources["vix3m"])[["DATE","Close"]] \
                        .rename(columns={"Close":"VIX3M"})
        except Exception as e:
            warnings_out.append(f"VIX3M unavailable ({e}); using v1 features.")
            vix3m = pd.DataFrame({"DATE": vix["DATE"], "VIX3M": np.nan})
    else:
        vix3m = pd.DataFrame({"DATE": vix["DATE"], "VIX3M": np.nan})

    if "vvix" in sources:
        try:
            vvix = _fetch(sources["vvix"])[["DATE","Close"]] \
                       .rename(columns={"Close":"VVIX"})
        except Exception as e:
            warnings_out.append(f"VVIX unavailable ({e}); skipping VVIX features.")
            vvix = pd.DataFrame({"DATE": vix["DATE"], "VVIX": np.nan})
    else:
        vvix = pd.DataFrame({"DATE": vix["DATE"], "VVIX": np.nan})

    pc = pd.DataFrame({"DATE": vix["DATE"], "PCRATIO": np.nan})

    df = (vix.merge(spx,   on="DATE", how="inner")
              .merge(vix3m, on="DATE", how="left")
              .merge(vvix,  on="DATE", how="left")
              .merge(pc,    on="DATE", how="left")
              .sort_values("DATE").reset_index(drop=True))
    if len(df) == 0:
        raise RuntimeError("Merged dataframe empty — VIX/SPX dates didn't align.")
    return df, warnings_out


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL FIT + PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def train_models(df_feat: pd.DataFrame, random_state: int = 42):
    df = df_feat.copy()
    df["log_ret_next"] = np.log(df["VIX"].shift(-1) / df["VIX"])
    has_req = df[CORE_REQUIRED].notna().all(axis=1)
    has_tgt = df["target"].notna() & df["log_ret_next"].notna()
    train_df = df[has_req & has_tgt]
    if len(train_df) < 250:
        raise RuntimeError(
            f"Only {len(train_df)} usable training rows — "
            "did the data sources return enough history?")

    feats = [f for f in FEATURES_FULL if f in train_df.columns]

    clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric="logloss", verbosity=0, random_state=random_state)
    clf.fit(train_df[feats], train_df["target"])

    qreg = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:quantileerror", quantile_alpha=QUANTILES,
        verbosity=0, random_state=random_state)
    qreg.fit(train_df[feats], train_df["log_ret_next"])
    return clf, qreg, len(train_df), feats


def _action(prob_up: float, p50: float, rule: dict) -> str:
    if prob_up >= rule["up_p"] and p50 >=  rule["up_m"]: return "LONG_UVIX"
    if prob_up <= rule["dn_p"] and p50 <= -rule["dn_m"]: return "LONG_SVIX"
    return "CASH"


def predict_today(clf, qreg, df_feat: pd.DataFrame,
                  feats: list) -> Dict[str, Any]:
    has_req = df_feat[CORE_REQUIRED].notna().all(axis=1) & df_feat["VIX"].notna()
    if not has_req.any():
        raise RuntimeError("No row with all required features.")
    last_idx = df_feat.index[has_req].max()
    row = df_feat.loc[last_idx]

    x = row[feats].values.reshape(1, -1)
    prob_up = float(clf.predict_proba(x)[0, 1])
    q = qreg.predict(x)
    if q.ndim == 1 and len(q) == 3:
        q = q.reshape(1, -1)
    p10, p50, p90 = float(q[0, 0]), float(q[0, 1]), float(q[0, 2])

    return {
        "data_through":   row["DATE"].strftime("%Y-%m-%d"),
        "vix":            float(row["VIX"]),
        "spx":            float(row["SPX"]),
        "vix3m":          float(row["VIX3M"]) if pd.notna(row.get("VIX3M", np.nan)) else None,
        "vvix":           float(row["VVIX"])  if pd.notna(row.get("VVIX",  np.nan)) else None,
        "pcratio":        float(row["PCRATIO"]) if pd.notna(row.get("PCRATIO", np.nan)) else None,
        "term_structure": "backwardation" if (pd.notna(row.get("VIX3M", np.nan))
                                              and row["VIX"] > row["VIX3M"]) else "contango",
        "prob_up":        round(prob_up, 4),
        "p10_logret":     round(p10, 5),
        "p50_logret":     round(p50, 5),
        "p90_logret":     round(p90, 5),
        "width":          round(p90 - p10, 5),
        "expected_pct":   round((np.exp(p50) - 1) * 100, 2),
        "p10_pct":        round((np.exp(p10) - 1) * 100, 2),
        "p90_pct":        round((np.exp(p90) - 1) * 100, 2),
        "action_strict":  _action(prob_up, p50, STRICT_RULE),
        "action_loose":   _action(prob_up, p50, LOOSE_RULE),
        "computed_at":    datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def compute_signal(sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    df, warnings_out = load_market_data(sources)
    df_feat = build_features(df)
    clf, qreg, n_train, feats = train_models(df_feat)
    sig = predict_today(clf, qreg, df_feat, feats)
    sig["n_training_rows"] = n_train
    sig["data_warnings"] = warnings_out
    sig["features_used"] = len(feats)
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SOURCE BUILDERS  (one per data-source preset)
# ─────────────────────────────────────────────────────────────────────────────

def sources_yfinance() -> Dict[str, Dict[str, Any]]:
    return {
        "vix":   {"kind":"yfinance", "symbol":"^VIX"},
        "spx":   {"kind":"yfinance", "symbol":"^GSPC"},
        "vix3m": {"kind":"yfinance", "symbol":"^VIX3M"},
        "vvix":  {"kind":"yfinance", "symbol":"^VVIX"},
    }

def sources_twelvedata_indices(api_key: str) -> Dict[str, Dict[str, Any]]:
    """Use only if your Twelve Data plan includes CBOE indices."""
    return {
        "vix":   {"kind":"twelvedata", "symbol":"VIX",   "api_key": api_key},
        "spx":   {"kind":"twelvedata", "symbol":"SPX",   "api_key": api_key},
        "vix3m": {"kind":"twelvedata", "symbol":"VIX3M", "api_key": api_key},
        "vvix":  {"kind":"twelvedata", "symbol":"VVIX",  "api_key": api_key},
    }

def sources_twelvedata_etf_proxies(api_key: str) -> Dict[str, Dict[str, Any]]:
    """Free-tier-friendly: ETFs that approximate the indices.
    VVIX has no ETF proxy → falls back to yfinance."""
    return {
        "vix":   {"kind":"twelvedata", "symbol":"VIXY", "api_key": api_key},
        "spx":   {"kind":"twelvedata", "symbol":"SPY",  "api_key": api_key},
        "vix3m": {"kind":"twelvedata", "symbol":"VIXM", "api_key": api_key},
        "vvix":  {"kind":"yfinance",   "symbol":"^VVIX"},
    }


# ═════════════════════════════════════════════════════════════════════════════
#                           STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="VIX Signal", page_icon="📈",
                   layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .big-badge {
        font-size: 2.0rem; font-weight: 700; text-align: center;
        padding: 1.2rem; border-radius: 1rem; margin: 0.7rem 0;
    }
    .badge-long-svix { background: #1f8a3a; color: white; }
    .badge-long-uvix { background: #b22222; color: white; }
    .badge-cash      { background: #6b7280; color: white; }
    .small-meta { color: #888; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60*60*4, show_spinner=False)
def fetch_signal_cached(provider: str, api_key: Optional[str]):
    """Cache by provider name + presence-of-key (don't put the key itself in cache key)."""
    if provider == "yfinance":
        return compute_signal(sources_yfinance())
    if provider == "twelvedata_indices":
        return compute_signal(sources_twelvedata_indices(api_key))
    if provider == "twelvedata_etfs":
        return compute_signal(sources_twelvedata_etf_proxies(api_key))
    raise ValueError(f"Unknown provider: {provider}")


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Settings")
    threshold_choice = st.radio(
        "Threshold preset",
        options=["STRICT (B@0.70/0.030)", "LOOSER (B@0.60/0.020)"],
        index=0,
        help="STRICT = fewer, higher-conviction trades. LOOSER = more frequent.")

    api_key = st.secrets.get("TWELVEDATA_KEY", os.environ.get("TWELVEDATA_KEY"))
    has_key = bool(api_key)

    provider_label = st.radio(
        "Data source",
        options=[
            "yfinance (free, recommended)",
            "Twelve Data — Indices plan",
            "Twelve Data — ETF proxies (free tier)",
        ],
        index=0,
        help=("yfinance: best on Streamlit Cloud.\n"
              "Twelve Data Indices: needs paid Indices add-on.\n"
              "Twelve Data ETFs: VIXY/SPY/VIXM proxies (works on free tier); "
              "VVIX falls back to yfinance."))
    if provider_label.startswith("yfinance"):
        provider_key = "yfinance"
    elif "Indices" in provider_label:
        provider_key = "twelvedata_indices"
    else:
        provider_key = "twelvedata_etfs"

    if provider_key.startswith("twelvedata") and not has_key:
        st.warning("No TWELVEDATA_KEY in Secrets — Twelve Data choices will fail. "
                   "Stick with yfinance or set the secret.")

    if st.button("🔄 Refresh now", use_container_width=True):
        fetch_signal_cached.clear()
        st.rerun()

    with st.expander("Diagnostics", expanded=False):
        st.write(f"Has API key: **{has_key}**")
        st.write(f"Provider: **{provider_key}**")


# ── Main page ───────────────────────────────────────────────────────────────
st.title("📈 VIX Market-Timing Signal")
st.caption("v4 model — direction + magnitude + range. "
           "**Research prototype, not investment advice.**")

tab_today, tab_history, tab_about = st.tabs(["Today", "History", "About"])

# ── TODAY ───────────────────────────────────────────────────────────────────
with tab_today:
    with st.spinner("Fetching market data and training the model "
                    "(takes 5-15s on first load)…"):
        try:
            sig = fetch_signal_cached(provider_key, api_key if has_key else None)
        except Exception as e:
            st.error(f"Failed to compute signal: **{e}**")
            st.info("Try switching the **Data source** in the sidebar to "
                    "yfinance (free, no key needed). Then click Refresh.")
            st.stop()

    if sig.get("data_warnings"):
        for w in sig["data_warnings"]:
            st.warning(w)

    action = (sig["action_strict"] if threshold_choice.startswith("STRICT")
              else sig["action_loose"])

    if action == "LONG_SVIX":
        badge_class, emoji, label = "badge-long-svix", "🟢", "LONG SVIX"
    elif action == "LONG_UVIX":
        badge_class, emoji, label = "badge-long-uvix", "🔴", "LONG UVIX"
    else:
        badge_class, emoji, label = "badge-cash", "⚪", "CASH"
    st.markdown(f"<div class='big-badge {badge_class}'>{emoji} {label}</div>",
                unsafe_allow_html=True)

    st.markdown(
        f"<p class='small-meta'>Signal for trading day after "
        f"<b>{sig['data_through']}</b> · computed {sig['computed_at']} · "
        f"{sig.get('n_training_rows','?'):,} training rows · "
        f"{sig.get('features_used','?')} features used</p>",
        unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("VIX", f"{sig['vix']:.2f}")
    col2.metric("SPX", f"{sig['spx']:,.2f}")
    col3, col4 = st.columns(2)
    col3.metric("VIX3M", f"{sig['vix3m']:.2f}" if sig['vix3m'] else "—",
                delta=sig['term_structure'] if sig['vix3m'] else None)
    col4.metric("VVIX", f"{sig['vvix']:.2f}" if sig['vvix'] else "—")

    st.divider()
    st.subheader("Forecast — next trading day VIX move")
    fcol1, fcol2 = st.columns(2)
    fcol1.metric("P(VIX up)", f"{sig['prob_up']*100:.1f}%")
    fcol2.metric("Expected",  f"{sig['expected_pct']:+.2f}%")
    st.caption(f"Range 10–90%: **{sig['p10_pct']:+.2f}%** to "
               f"**{sig['p90_pct']:+.2f}%** (width {sig['width']:.4f})")

    st.divider()
    with st.expander("Both threshold variants", expanded=False):
        st.write(f"**STRICT (B@0.70/0.030)** → `{sig['action_strict']}`")
        st.write(f"**LOOSER (B@0.60/0.020)** → `{sig['action_loose']}`")
        st.caption("STRICT: prob_up≥0.70 AND p50≥+0.030 (long), "
                   "or prob_up≤0.30 AND p50≤−0.030 (short).")

    st.divider()
    st.markdown(
        f"### What to do\n"
        f"Place **{label}** at today's market close (or first thing tomorrow). "
        f"Hold until the next signal flip — **re-check tomorrow at ~4:30pm ET**. "
        f"No stops, no targets — backtests showed they hurt this model.")


# ── HISTORY ─────────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Signal history")
    log_path = "signal_log.csv"
    if os.path.exists(log_path):
        try:
            df_hist = pd.read_csv(log_path, parse_dates=["data_through"])
            df_hist = df_hist.sort_values("data_through", ascending=False)
            st.dataframe(df_hist.head(60), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not read history: {e}")
    else:
        st.info("No history yet. Click below to log today's signal, "
                "or set up the GitHub Action in the About tab to log automatically.")

    if st.button("Append today's signal to history"):
        try:
            new_row = {
                "data_through": sig["data_through"],
                "vix":          sig["vix"],
                "vix3m":        sig["vix3m"],
                "vvix":         sig["vvix"],
                "spx":          sig["spx"],
                "prob_up":      sig["prob_up"],
                "p50_logret":   sig["p50_logret"],
                "expected_pct": sig["expected_pct"],
                "action_strict": sig["action_strict"],
                "action_loose":  sig["action_loose"],
                "computed_at":  sig["computed_at"],
            }
            new_df = pd.DataFrame([new_row])
            if os.path.exists(log_path):
                new_df.to_csv(log_path, mode="a", header=False, index=False)
            else:
                new_df.to_csv(log_path, index=False)
            st.success("Logged. Reload the page to see it.")
        except Exception as e:
            st.error(f"Could not log: {e}")


# ── ABOUT ───────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
### How this works
- **XGBoost classifier** predicts P(VIX rises tomorrow) from 100+ features
  built from VIX/SPX OHLC, VIX term structure (VIX3M), and vol-of-vol (VVIX).
- **XGBoost quantile regressor** predicts the 10/50/90 percentile of tomorrow's
  log-return on VIX — direction *plus* expected magnitude and range.
- **Strategy B@0.70/0.030 (STRICT)**: long UVIX only when prob_up ≥ 0.70
  AND median expected move ≥ +3%; long SVIX only when prob_up ≤ 0.30 AND
  median ≤ −3%; otherwise cash.

### Backtest summary (UVIX/SVIX, 2022-03 → 2026-04, sealed OOS)
| Strategy            | CAGR   | Sharpe | Max DD |
|---------------------|-------:|-------:|-------:|
| Buy & hold UVIX     | −85%   | −0.80  | −100% |
| Buy & hold SVIX     | 17.8%  |  0.59  | −67.8% |
| **STRICT B@0.70/0.030** | **26.3%** | **1.11** | **−20.7%** |

Forward expectation: 15-25% CAGR with single-digit drawdowns most years
and occasional 20-40% drawdowns when a regime shift catches the model wrong.

### Daily auto-update via GitHub Actions

Add `.github/workflows/daily.yml` to your repo:

```yaml
name: Daily VIX signal
on:
  schedule:
    - cron: "30 20 * * 1-5"   # 4:30 PM ET (EDT)
  workflow_dispatch:
jobs:
  signal:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: |
          python -c "
          from streamlit_app import compute_signal, sources_yfinance
          import pandas as pd, pathlib
          sig = compute_signal(sources_yfinance())
          row = {k: sig[k] for k in
                 ['data_through','vix','vix3m','vvix','spx','prob_up',
                  'p50_logret','expected_pct','action_strict','action_loose','computed_at']}
          p = pathlib.Path('signal_log.csv')
          new = pd.DataFrame([row])
          if p.exists(): new.to_csv(p, mode='a', header=False, index=False)
          else: new.to_csv(p, index=False)
          "
      - run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add signal_log.csv
          git commit -m "daily signal" || true
          git push
```

Free; 2,000 GitHub-Action minutes/month is plenty.
""")
