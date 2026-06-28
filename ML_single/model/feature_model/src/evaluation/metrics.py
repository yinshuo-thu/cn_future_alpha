import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_ic(pred, label):
    """Pooled cosine-similarity IC (per spec: mean(a*y)/sqrt(mean(a*a)*mean(y*y)))"""
    pred = np.asarray(pred, dtype=float)
    label = np.asarray(label, dtype=float)
    mask = ~(np.isnan(pred) | np.isnan(label))
    if mask.sum() < 2:
        return np.nan
    a = pred[mask]
    y = label[mask]
    num = np.mean(a * y)
    denom = np.sqrt(np.mean(a * a) * np.mean(y * y))
    if denom < 1e-12:
        return np.nan
    return float(num / denom)


def ic_by_period(df, pred_col="pred", label_col="label", period="Y"):
    df = df.copy().dropna(subset=[pred_col, label_col])
    if period == "Y":
        df["_period"] = df["datetime"].dt.year
    else:
        df["_period"] = df["datetime"].dt.to_period("M").astype(str)
    results = {}
    for p, grp in df.groupby("_period"):
        results[p] = compute_ic(grp[pred_col].values, grp[label_col].values)
    return pd.Series(results).sort_index()


def ic_summary(df, pred_col="pred", label_col="label", title="", save_dir=None):
    ic_total = compute_ic(df[pred_col].values, df[label_col].values)
    by_year = ic_by_period(df, pred_col, label_col, "Y")
    by_month = ic_by_period(df, pred_col, label_col, "M")
    ir = by_month.mean() / by_month.std() if by_month.std() > 0 else np.nan
    print(f"\n{'='*50}")
    if title:
        print(f"  {title}")
    print(f"  Total IC:   {ic_total:.4f}")
    print(f"  Monthly IC  mean={by_month.mean():.4f}  std={by_month.std():.4f}  IR={ir:.4f}")
    print(f"\nYearly IC:")
    print(by_year.to_string())
    print("=" * 50)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_monthly_ic(by_month, title=title, save_path=os.path.join(save_dir, f"monthly_ic_{title.replace(' ','_')}.png"))
        by_year.to_csv(os.path.join(save_dir, f"yearly_ic_{title.replace(' ','_')}.csv"))
    return ic_total, by_year, by_month


def plot_monthly_ic(by_month, title="Monthly IC", save_path=None):
    fig, ax = plt.subplots(figsize=(14, 4))
    by_month.plot(kind="bar", ax=ax, color="steelblue", alpha=0.7)
    ax.axhline(0, color="red", linewidth=0.8)
    ax.axhline(by_month.mean(), color="green", linewidth=1, linestyle="--",
               label=f"mean={by_month.mean():.4f}")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
    return fig


def sanity_check():
    np.random.seed(42)
    n = 10000
    y = np.random.randn(n)
    ic_perfect = compute_ic(y, y)
    ic_random = compute_ic(np.random.randn(n), y)
    ic_flip = compute_ic(-y, y)
    assert abs(ic_perfect - 1.0) < 0.001, f"perfect pred IC={ic_perfect}, expected ~1"
    assert abs(ic_random) < 0.05, f"random IC={ic_random}, expected ~0"
    assert abs(ic_flip - (-1.0)) < 0.001, f"flipped IC={ic_flip}, expected ~-1"
    print(f"[Evaluator sanity] perfect={ic_perfect:.4f} random={ic_random:.4f} flip={ic_flip:.4f} => PASSED")


if __name__ == "__main__":
    sanity_check()
