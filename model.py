"""模型:LightGBM 排序回归 + ICIR 加权线性合成;Walk-forward(带 embargo)防未来函数。"""
import warnings

import numpy as np
import pandas as pd

import config
from factors import Dataset

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except ImportError:  # 没装也能跑,退化为线性
    lgb = None


def ic_matrix(ds: Dataset):
    """每个交易日、每个因子与未来收益排名的截面相关(Pearson on rank≈RankIC)。"""
    X, y, valid = ds.X, ds.y, ds.valid
    D, n, F = X.shape
    ic = np.full((D, F), np.nan)
    for f in range(F):
        x = X[:, :, f]
        m = valid & ~np.isnan(x) & ~np.isnan(y)
        cnt = m.sum(1)
        mx = np.where(m, x, 0).sum(1) / np.maximum(cnt, 1)
        my = np.where(m, y, 0).sum(1) / np.maximum(cnt, 1)
        dx = np.where(m, x - mx[:, None], 0)
        dy = np.where(m, y - my[:, None], 0)
        den = np.sqrt((dx ** 2).sum(1) * (dy ** 2).sum(1))
        ic[:, f] = np.where((cnt > 30) & (den > 0), (dx * dy).sum(1) / np.where(den > 0, den, 1), np.nan)
    return ic


def _fit_predict(ds, ic, a, b, i, j):
    """用 [a,b) 训练,预测 [i,j)。返回 (j-i, n) 的得分(0~1 分位),不可交易为 NaN。"""
    F = ds.X.shape[2]
    Xtr = ds.X[a:b].reshape(-1, F)
    ytr = ds.y[a:b].reshape(-1)
    mtr = ds.valid[a:b].reshape(-1) & ~np.isnan(ytr)

    ic_tr = ic[a:b]
    mu, sd = np.nanmean(ic_tr, 0), np.nanstd(ic_tr, 0)
    w = np.nan_to_num(mu / np.where(sd > 0, sd, np.nan))          # ICIR 权重(带方向)
    Xte = ds.X[i:j]
    lin = np.nan_to_num(Xte) @ w                                   # (j-i, n)

    s_lin = pd.DataFrame(np.where(ds.valid[i:j], lin, np.nan)).rank(axis=1, pct=True)
    if lgb is None or mtr.sum() < 5000:
        return s_lin.values

    params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
                  bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, lambda_l2=5.0,
                  verbose=-1, num_threads=0)
    booster = lgb.train(params, lgb.Dataset(Xtr[mtr], ytr[mtr]), num_boost_round=300)   # 原生接口,不依赖 sklearn
    pred = booster.predict(Xte.reshape(-1, F)).reshape(j - i, -1)
    s_lgb = pd.DataFrame(np.where(ds.valid[i:j], pred, np.nan)).rank(axis=1, pct=True)
    out = config.BLEND_LGB * s_lgb.values + (1 - config.BLEND_LGB) * s_lin.values
    return out


def walk_forward(ds: Dataset, eval_days=None, step=20, window=300, embargo=2, min_train=150):
    """滚动样本外预测。训练集截止到 i-H-embargo,保证标签不包含预测日之后的信息。"""
    H = config.HORIZON
    D, n, _ = ds.X.shape
    ic = ic_matrix(ds)
    start = min_train + H + embargo
    if eval_days:
        start = max(start, D - eval_days)
    out = np.full((D, n), np.nan)
    for i in range(start, D, step):
        b = i - H - embargo
        a = max(0, b - window)
        if b - a < 100:
            continue
        j = min(i + step, D)
        out[i:j] = _fit_predict(ds, ic, a, b, i, j)
    return pd.DataFrame(out, index=ds.dates, columns=ds.codes)


def predict_latest(ds: Dataset, window=300, embargo=2):
    """用最新可得标签训练,对最后一个交易日打分(0~1,越高越看好)。"""
    H = config.HORIZON
    D = ds.X.shape[0]
    ic = ic_matrix(ds)
    i = D - 1
    b = i - H - embargo
    a = max(0, b - window)
    s = _fit_predict(ds, ic, a, b, i, D)[0]
    return pd.Series(s, index=ds.codes, name="score")


def eval_oos(score: pd.DataFrame, ds: Dataset, top_n=None):
    """样本外评价:RankIC、TopN 超额收益、胜率。"""
    top_n = top_n or config.TOP_N
    rows = []
    for d in range(len(ds.dates)):
        s = score.iloc[d]
        f = pd.Series(ds.fwd[d], index=ds.codes)
        m = s.notna() & f.notna() & ds.valid[d]
        if m.sum() < 50:
            continue
        s, f = s[m], f[m]
        ic = s.rank().corr(f.rank())
        top = s.nlargest(top_n).index
        rows.append((ds.dates[d], ic, f[top].mean() - f.mean(), f[top].mean()))
    r = pd.DataFrame(rows, columns=["date", "rank_ic", "top_excess", "top_ret"]).set_index("date")
    if r.empty:
        return r, {}
    stat = dict(
        rank_ic_mean=r["rank_ic"].mean(),
        icir=r["rank_ic"].mean() / (r["rank_ic"].std() + 1e-12),
        ic_pos_ratio=(r["rank_ic"] > 0).mean(),
        top_excess_mean=r["top_excess"].mean(),
        top_win_rate=(r["top_excess"] > 0).mean(),
        n_days=len(r),
    )
    return r, stat


def factor_report(ds: Dataset, last_n=250):
    """各因子近 N 日 IC 统计,用于监控因子衰减。"""
    ic = ic_matrix(ds)[-last_n:]
    df = pd.DataFrame({"IC均值": np.nanmean(ic, 0), "ICIR": np.nanmean(ic, 0) / (np.nanstd(ic, 0) + 1e-12)},
                      index=ds.feats)
    return df.sort_values("ICIR", key=abs, ascending=False)

def explain_latest(ds: Dataset, code, window=300, embargo=2):
    """单只股票:最新模型分位、全市场排名、因子贡献(LightGBM SHAP + ICIR线性贡献)。"""
    if code not in ds.codes:
        return None
    H = config.HORIZON
    D, n, F = ds.X.shape
    j = ds.codes.get_loc(code)
    ic = ic_matrix(ds)
    i = D - 1
    b = i - H - embargo
    a = max(0, b - window)
    ic_tr = ic[a:b]
    mu, sd = np.nanmean(ic_tr, 0), np.nanstd(ic_tr, 0)
    w = np.nan_to_num(mu / np.where(sd > 0, sd, np.nan))
    X, valid = ds.X[i], ds.valid[i]
    lin = np.nan_to_num(X) @ w
    lin_pct = pd.Series(np.where(valid, lin, np.nan)).rank(pct=True).values
    res = dict(valid=bool(valid[j]), n_valid=int(valid.sum()), feats=ds.feats, z=X[j].astype(float),
               lin_contrib=np.nan_to_num(X[j]).astype(float) * w, lin_pct=float(lin_pct[j]) if valid[j] else np.nan,
               date=ds.dates[i], shap=None, lgb_pct=np.nan)
    blend = lin_pct
    if lgb is not None:
        Xtr = ds.X[a:b].reshape(-1, F)
        ytr = ds.y[a:b].reshape(-1)
        mtr = ds.valid[a:b].reshape(-1) & ~np.isnan(ytr)
        if mtr.sum() >= 5000:
            params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
                          bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, lambda_l2=5.0,
                          verbose=-1, num_threads=0)
            booster = lgb.train(params, lgb.Dataset(Xtr[mtr], ytr[mtr]), num_boost_round=300)
            pred = booster.predict(X)
            lgb_pct = pd.Series(np.where(valid, pred, np.nan)).rank(pct=True).values
            blend = config.BLEND_LGB * lgb_pct + (1 - config.BLEND_LGB) * lin_pct
            res["shap"] = booster.predict(X[j:j + 1].astype(float), pred_contrib=True)[0][:F]
            res["lgb_pct"] = float(lgb_pct[j]) if valid[j] else np.nan
    res["score"] = float(blend[j]) if valid[j] else np.nan
    if valid[j]:
        res["rank"] = int((blend[valid] > blend[j]).sum() + 1)
    return res
