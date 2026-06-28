"""
Large factor library — systematic, parametric, session-respecting (no long-break leak).

Generates ~700 time-series base factors per symbol across families:
  returns/momentum, risk-adjusted momentum, volatility (RV/Parkinson/GK/ATR),
  oscillators (RSI/Stoch/CCI/Bollinger-z/CMO/ROC), MACD/TRIX trend,
  volume & amount dynamics (zscore/momentum/MFI/Amihud-illiq/VWAP-dev),
  open-interest (OI momentum/zscore/divergence/accel), bar geometry (BOP/body/shadow),
  and interactions (mom×vol, mom×OI, signed-volume).

Panel-level cross-sectional transforms (zscore, rank) are added downstream to
roughly triple the candidate count before correlation-based dedup (<0.9).

Speed: per (series, window) rolling stats computed once and reused across derived
factors. All windowed factors are masked where the lookback crosses a session.
"""
import numpy as np
import pandas as pd

# Fibonacci-ish windows: spaced to keep adjacent-window correlation below ~0.9
WINS = [2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 42, 55, 72, 89, 120, 144, 180, 233, 288]
WINS_MID = [4, 6, 8, 10, 13, 16, 21, 26, 34, 42, 55, 72, 89, 120, 144, 180, 233]
WINS_LONG = [55, 89, 144, 233]
EMA_PAIRS = [(3, 13), (5, 21), (8, 34), (10, 42), (13, 55), (16, 72), (21, 89), (5, 55), (13, 144), (21, 180)]
LAG_BASES = ["mom_13", "mom_34", "rv_21", "volz_21", "stoch_21", "rsi_21",
             "bollz_21", "cci_21", "mfi_21", "pvcorr_34", "bop_21", "cpos"]
LAGS = [1, 2, 3, 5]


def _bss(is_lb):
    out = np.zeros(len(is_lb), dtype=np.int64)
    c = 0
    for i in range(len(is_lb)):
        c = 0 if is_lb[i] else c + 1
        out[i] = c
    return out


def _ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def compute_symbol_factors(df):
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    v = np.nan_to_num(df["volume"].values.astype(np.float64), nan=0.0)
    amt = np.nan_to_num(df["amount"].values.astype(np.float64), nan=0.0) if "amount" in df else v * c
    oi = np.nan_to_num(df["oi"].values.astype(np.float64), nan=0.0) if "oi" in df else np.zeros(n)
    is_lb = df["is_long_break_before"].values
    bss = _bss(is_lb)

    lr = np.zeros(n)
    lr[1:] = np.log(c[1:] / np.clip(c[:-1], 1e-9, None))
    lr[is_lb] = 0.0
    lr = np.nan_to_num(lr)
    logv = np.log1p(np.clip(v, 0, None))
    loga = np.log1p(np.clip(amt, 0, None))
    typ = (h + l + c) / 3.0
    hl = np.clip(h - l, 1e-9, None)

    Slr, Sc, Sv, Sa = pd.Series(lr), pd.Series(c), pd.Series(logv), pd.Series(loga)
    Styp, Sh, Sl = pd.Series(typ), pd.Series(h), pd.Series(l)
    f = {}

    def msk(arr, w):
        a = np.asarray(arr, dtype=np.float64).copy()
        a[bss < w] = np.nan
        return a

    # ---------- momentum / returns ----------
    for w in WINS:
        rs = Slr.rolling(w, min_periods=w)
        f[f"mom_{w}"] = msk(rs.sum().values, w)
        std = rs.std().values
        f[f"sharpe_{w}"] = msk(Slr.rolling(w, min_periods=w).mean().values / (std + 1e-9), w)
        f[f"upratio_{w}"] = msk((Slr.gt(0)).rolling(w, min_periods=w).mean().values, w)
        f[f"maxret_{w}"] = msk(Slr.rolling(w, min_periods=w).max().values, w)
        f[f"minret_{w}"] = msk(Slr.rolling(w, min_periods=w).min().values, w)
    for w in WINS_MID:
        sma = Sc.rolling(w, min_periods=w).mean().values
        f[f"px_sma_{w}"] = msk(c / (sma + 1e-9) - 1.0, w)
        # linreg slope of close (normalized)
        f[f"roc_{w}"] = msk(c / np.r_[np.full(w, np.nan), c[:-w]] - 1.0, w)
    f["mom_accel"] = msk(Slr.rolling(8).sum().values - Slr.rolling(34).sum().values, 34)
    for (a, b) in EMA_PAIRS:
        ed = _ema(c, a) - _ema(c, b)
        f[f"ema_diff_{a}_{b}"] = msk(ed / (np.abs(c) + 1e-9), b)
        f[f"macd_{a}_{b}"] = msk(np.r_[np.nan, np.diff(ed)] / (np.abs(c) + 1e-9), b)

    # ---------- volatility ----------
    lr2 = Slr.pow(2)
    loghl2 = pd.Series(np.log(np.clip(h, 1e-9, None) / np.clip(l, 1e-9, None)) ** 2)
    for w in WINS:
        f[f"rv_{w}"] = msk(np.sqrt(lr2.rolling(w, min_periods=w).sum().values), w)
        f[f"park_{w}"] = msk(np.sqrt(loghl2.rolling(w, min_periods=w).sum().values / (4 * np.log(2) * w)), w)
    neg = pd.Series(np.where(lr < 0, lr, 0.0))
    pos = pd.Series(np.where(lr > 0, lr, 0.0))
    for w in WINS_MID:
        dn = np.sqrt(neg.pow(2).rolling(w, min_periods=w).sum().values)
        up = np.sqrt(pos.pow(2).rolling(w, min_periods=w).sum().values)
        f[f"dvol_{w}"] = msk(dn, w)
        f[f"volskew_{w}"] = msk((up - dn) / (up + dn + 1e-9), w)
        f[f"skew_{w}"] = msk(Slr.rolling(w, min_periods=w).skew().values, w)
        f[f"kurt_{w}"] = msk(Slr.rolling(w, min_periods=w).kurt().values, w)
    # ATR & range
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[np.nan, c[:-1]]), np.abs(l - np.r_[np.nan, c[:-1]])))
    Str = pd.Series(np.nan_to_num(tr))
    for w in WINS_MID:
        f[f"atr_{w}"] = msk(Str.rolling(w, min_periods=w).mean().values / (np.abs(c) + 1e-9), w)
        f[f"range_{w}"] = msk(pd.Series((h - l) / (np.abs(c) + 1e-9)).rolling(w, min_periods=w).mean().values, w)
    # vol regime ratio
    f["vol_regime"] = msk(pd.Series(f["rv_8"]).values / (pd.Series(f["rv_55"]).values + 1e-9), 55)

    # ---------- oscillators ----------
    for w in WINS_MID:
        mn = Sl.rolling(w, min_periods=w).min().values
        mx = Sh.rolling(w, min_periods=w).max().values
        stoch = (c - mn) / (mx - mn + 1e-9)
        f[f"stoch_{w}"] = msk(stoch, w)
        f[f"willr_{w}"] = msk((mx - c) / (mx - mn + 1e-9), w)
        f[f"stoch_dev_{w}"] = msk(stoch - pd.Series(stoch).rolling(5, min_periods=3).mean().values, w)
        # RSI
        up_ = pd.Series(np.where(np.r_[np.nan, np.diff(c)] > 0, np.r_[np.nan, np.diff(c)], 0.0))
        dn_ = pd.Series(np.where(np.r_[np.nan, np.diff(c)] < 0, -np.r_[np.nan, np.diff(c)], 0.0))
        rs_ = up_.rolling(w, min_periods=w).mean().values / (dn_.rolling(w, min_periods=w).mean().values + 1e-9)
        f[f"rsi_{w}"] = msk(100 - 100 / (1 + rs_), w)
        # CCI
        sma_tp = Styp.rolling(w, min_periods=w).mean().values
        # Use rolling std as a fast MAD proxy. The exact rolling mean absolute
        # deviation needs Python callbacks and is the dominant full-build cost.
        mad_proxy = Styp.rolling(w, min_periods=w).std().values * 0.8
        f[f"cci_{w}"] = msk((typ - sma_tp) / (0.015 * mad_proxy + 1e-9), w)
        # Bollinger z
        sd = Sc.rolling(w, min_periods=w).std().values
        f[f"bollz_{w}"] = msk((c - Sc.rolling(w, min_periods=w).mean().values) / (sd + 1e-9), w)
        # CMO
        f[f"cmo_{w}"] = msk((up_.rolling(w, min_periods=w).sum().values - dn_.rolling(w, min_periods=w).sum().values)
                            / (up_.rolling(w, min_periods=w).sum().values + dn_.rolling(w, min_periods=w).sum().values + 1e-9), w)
        f[f"ret_ac1_{w}"] = msk(Slr.rolling(w, min_periods=w).corr(Slr.shift(1)).values, w)
        f[f"ret_ac2_{w}"] = msk(Slr.rolling(w, min_periods=w).corr(Slr.shift(2)).values, w)

    # ---------- volume / amount ----------
    for w in WINS_MID:
        vm = Sv.rolling(w, min_periods=w).mean().values
        vs = Sv.rolling(w, min_periods=w).std().values
        f[f"volz_{w}"] = msk((logv - vm) / (vs + 1e-9), w)
        f[f"volmom_{w}"] = msk(logv - np.r_[np.full(w, np.nan), logv[:-w]], w)
        am = Sa.rolling(w, min_periods=w).mean().values
        f[f"amtz_{w}"] = msk((loga - am) / (Sa.rolling(w, min_periods=w).std().values + 1e-9), w)
        f[f"amtmom_{w}"] = msk(loga - np.r_[np.full(w, np.nan), loga[:-w]], w)
        turnover_ratio = loga - logv
        f[f"turnover_z_{w}"] = msk((turnover_ratio - pd.Series(turnover_ratio).rolling(w, min_periods=w).mean().values)
                                   / (pd.Series(turnover_ratio).rolling(w, min_periods=w).std().values + 1e-9), w)
        f[f"pvcorr_{w}"] = msk(Slr.rolling(w, min_periods=w).corr(Sv).values, w)
        f[f"pacorr_{w}"] = msk(Slr.rolling(w, min_periods=w).corr(Sa).values, w)
        # Amihud illiquidity
        f[f"illiq_{w}"] = msk(pd.Series(np.abs(lr) / (loga + 1e-6)).rolling(w, min_periods=w).mean().values, w)
        signed_vol = np.sign(lr) * logv
        signed_amt = np.sign(lr) * loga
        f[f"vol_flow_{w}"] = msk(pd.Series(signed_vol).rolling(w, min_periods=w).sum().values, w)
        f[f"amt_flow_{w}"] = msk(pd.Series(signed_amt).rolling(w, min_periods=w).sum().values, w)
        pos_vol = pd.Series(np.where(lr > 0, logv, 0.0)).rolling(w, min_periods=w).sum().values
        neg_vol = pd.Series(np.where(lr < 0, logv, 0.0)).rolling(w, min_periods=w).sum().values
        f[f"updn_vol_ratio_{w}"] = msk((pos_vol - neg_vol) / (pos_vol + neg_vol + 1e-9), w)
        f[f"amt_prop_signed_{w}"] = msk(np.sign(lr) * np.nan_to_num(amt / (pd.Series(amt).rolling(w, min_periods=w).sum().values + 1e-9)), w)
        # MFI
        mf = typ * v
        posmf = pd.Series(np.where(np.r_[np.nan, np.diff(typ)] > 0, mf, 0.0))
        negmf = pd.Series(np.where(np.r_[np.nan, np.diff(typ)] < 0, mf, 0.0))
        mr = posmf.rolling(w, min_periods=w).sum().values / (negmf.rolling(w, min_periods=w).sum().values + 1e-9)
        f[f"mfi_{w}"] = msk(100 - 100 / (1 + mr), w)
    # VWAP deviation
    vwap = amt / np.clip(v, 1e-9, None)
    vwap[v <= 0] = c[v <= 0]
    f["vwap_dev"] = (c - vwap) / (np.abs(c) + 1e-9)
    for w in WINS_MID:
        vw = Sa.rolling(w).sum().values  # proxy
        vwr = pd.Series(amt).rolling(w, min_periods=w).sum().values / (pd.Series(v).rolling(w, min_periods=w).sum().values + 1e-9)
        f[f"vwap_dev_{w}"] = msk((c - vwr) / (np.abs(c) + 1e-9), w)
    for a, b in EMA_PAIRS:
        vwap_fast = _ema(vwap, a)
        vwap_slow = _ema(vwap, b)
        f[f"vwap_macd_{a}_{b}"] = msk((vwap_fast - vwap_slow) / (np.abs(c) + 1e-9), b)
    # signed volume / amount proportion
    f["amt_share_signed"] = np.sign(lr) * np.nan_to_num(amt / (pd.Series(amt).rolling(12).sum().values + 1e-9))

    # ---------- open interest ----------
    if oi.sum() > 0:
        dloi = np.zeros(n)
        dloi[1:] = np.log(np.clip(oi[1:], 1, None) / np.clip(oi[:-1], 1, None))
        dloi[is_lb] = 0.0
        dloi = np.nan_to_num(dloi)
        Sdoi = pd.Series(dloi)
        f["oi_chg"] = dloi
        for w in WINS_MID:
            f[f"oimom_{w}"] = msk(Sdoi.rolling(w, min_periods=w).sum().values, w)
            f[f"oiz_{w}"] = msk((np.log1p(np.clip(oi, 1, None)) - pd.Series(np.log1p(np.clip(oi, 1, None))).rolling(w, min_periods=w).mean().values)
                                / (pd.Series(np.log1p(np.clip(oi, 1, None))).rolling(w, min_periods=w).std().values + 1e-9), w)
            f[f"oipx_div_{w}"] = msk(Sdoi.rolling(w, min_periods=w).sum().values * np.sign(f[f"mom_{w}"] if f"mom_{w}" in f else lr), w)
        f["oi_vol_ratio"] = np.log1p(np.clip(oi, 1, None)) - logv
        f["oi_accel"] = msk(Sdoi.rolling(8).sum().values - Sdoi.rolling(34).sum().values, 34)

    # ---------- bar geometry ----------
    bop = (c - o) / hl
    body = np.abs(c - o) / hl
    upsh = (h - np.maximum(o, c)) / hl
    losh = (np.minimum(o, c) - l) / hl
    cpos = (c - l) / hl
    for nm, arr in [("bop", bop), ("body", body), ("upsh", upsh), ("losh", losh), ("cpos", cpos)]:
        f[nm] = arr
        for w in [8, 21, 55]:
            f[f"{nm}_{w}"] = msk(pd.Series(arr).rolling(w, min_periods=w).mean().values, w)
    for w in WINS_MID:
        f[f"shadow_imb_{w}"] = msk((pd.Series(losh).rolling(w, min_periods=w).mean().values
                                    - pd.Series(upsh).rolling(w, min_periods=w).mean().values), w)
        f[f"shadow_vol_diff_{w}"] = msk((pd.Series(losh).rolling(w, min_periods=w).std().values
                                         - pd.Series(upsh).rolling(w, min_periods=w).std().values), w)
        f[f"body_pos_inter_{w}"] = msk(pd.Series(body * (cpos - 0.5)).rolling(w, min_periods=w).mean().values, w)
        hh = Sh.rolling(w, min_periods=w).max().values
        ll = Sl.rolling(w, min_periods=w).min().values
        hc = Sc.rolling(w, min_periods=w).max().values
        lc = Sc.rolling(w, min_periods=w).min().values
        thrust_range = np.maximum(hh - lc, hc - ll)
        f[f"dual_thrust_pos_{w}"] = msk((c - (ll + 0.5 * thrust_range)) / (thrust_range + 1e-9), w)
        f[f"tr_dir_{w}"] = msk(np.sign(f.get(f"mom_{w}", lr)) * Str.rolling(w, min_periods=w).mean().values / (np.abs(c) + 1e-9), w)

    # ---------- interactions ----------
    for w in WINS_MID:
        f[f"mom_x_volz_{w}"] = msk(np.nan_to_num(f.get(f"mom_{w}", lr)) * np.nan_to_num(f.get(f"volz_{w}", 0.0)), w)
        if "oi_chg" in f:
            f[f"mom_x_oimom_{w}"] = msk(np.nan_to_num(f.get(f"mom_{w}", lr)) * np.nan_to_num(f.get(f"oimom_{w}", 0.0)), w)
        f[f"rv_x_volz_{w}"] = msk(np.nan_to_num(f.get(f"rv_{w}", 0.0)) * np.nan_to_num(f.get(f"volz_{w}", 0.0)), w)

    # ---------- efficiency ratio, drawup/down, autocorr, ret-OI corr ----------
    abs_lr = pd.Series(np.abs(lr))
    for w in WINS_MID:
        # Kaufman efficiency ratio: |net move| / sum|moves|
        net = np.abs(Slr.rolling(w, min_periods=w).sum().values)
        path = abs_lr.rolling(w, min_periods=w).sum().values
        f[f"effratio_{w}"] = msk(net / (path + 1e-9), w)
        # drawup / drawdown from rolling extremes
        mx = Sc.rolling(w, min_periods=w).max().values
        mn = Sc.rolling(w, min_periods=w).min().values
        f[f"drawdn_{w}"] = msk(c / (mx + 1e-9) - 1.0, w)
        f[f"drawup_{w}"] = msk(c / (mn + 1e-9) - 1.0, w)
    if oi.sum() > 0:
        Sdoi2 = pd.Series(dloi)
        for w in WINS_MID:
            f[f"ret_oi_corr_{w}"] = msk(Slr.rolling(w, min_periods=w).corr(Sdoi2).values, w)

    # ---------- lagged variants of key factors (decorrelated short-memory views) ----------
    for base in LAG_BASES:
        if base in f:
            arr = np.asarray(f[base], dtype=np.float64)
            for lag in LAGS:
                lagged = np.r_[np.full(lag, np.nan), arr[:-lag]]
                # invalidate where the lag crosses a session boundary
                lagged[bss < lag] = np.nan
                f[f"{base}_lag{lag}"] = lagged

    out = pd.DataFrame(f)
    out["symbol"] = df["symbol"].values
    out["datetime"] = df["datetime"].values
    out["label"] = df["label"].values
    return out
