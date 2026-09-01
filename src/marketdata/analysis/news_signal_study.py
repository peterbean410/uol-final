"""
Offline study: does the EOD news snapshot carry forward-return signal for an FX pair?

Pure historical analysis - no RL, no training, no modelenv, no cluster. Reads the
cumulative EOD news snapshot and the D1 EOD price snapshot, builds a daily feature
panel, and measures each feature's information coefficient (IC) against forward
close-to-close returns, with a rotation null, a half-sample stability check and
quintile bucket returns.

Usage:
    python marketdata/analysis/news_signal_study.py
    python marketdata/analysis/news_signal_study.py --news s3://bucket/key.parquet
    python marketdata/analysis/news_signal_study.py --out-dir /tmp/newsstudy

Defaults resolve to the newest snapshot cached under ./s3data. Paths starting with
s3:// are read via boto3.

Timing convention (no look-ahead): a D1 bar labelled day d is taken to close at
d+1 00:00 UTC. Each news item is attributed to the first bar close at or after its
publication timestamp - the earliest moment it could have been acted on - and the
forward return is measured from that close onward.

Known limitation: weekend news is therefore attributed to Monday's bar and measured
from Monday's *close*, which discards the Friday-close to Monday-close gap. Any
weekend-news effect is understated by this panel.
"""

import argparse
import glob
import io
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

SENTIMENT_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

TA_BOILERPLATE_SOURCES = {"action forex", "dailyfx"}

TOPIC_FLAGS = {
    "flag_fed": r"\b(?:fed|fed's|federal reserve|fomc|powell)\b",
    "flag_cpi": r"\b(?:cpi|inflation)\b",
    "flag_nfp": r"\b(?:nfp|non-farm|nonfarm|payroll|payrolls)\b",
    "flag_boj": r"\b(?:boj|boj's|bank of japan|ueda|kuroda)\b",
}

HORIZONS = (1, 5)
N_ROTATIONS = 1000
ROTATION_SEED = 20260725
IC_MIN_ABS = 0.05
ALPHA = 0.05
DEFAULT_SPREAD_PIPS = 1.0
PIP_SIZE_JPY = 0.01

FEATURES = (
    "sent_mean",
    "sent_mean_core",
    "sent_disp",
    "n_articles",
    "n_articles_z20",
    "sent_mean_5d",
    "sent_mean_20d",
    *TOPIC_FLAGS.keys(),
)


@dataclass
class FeatureResult:
    """IC and stability statistics for one (feature, horizon) pair."""

    feature: str
    horizon: int
    n: int
    ic: float
    ic_spearman: float
    p_rotation: float
    noise_band: float
    ic_first_half: float
    ic_second_half: float
    spread: float
    buckets: pd.DataFrame

    @property
    def sign_stable(self) -> bool:
        return np.sign(self.ic_first_half) == np.sign(self.ic_second_half) != 0

    @property
    def passes(self) -> bool:
        return (
            abs(self.ic) >= max(IC_MIN_ABS, 2 * self.noise_band)
            and self.p_rotation < ALPHA
            and self.sign_stable
        )


def _read_parquet(path: str) -> pd.DataFrame:
    """Read a parquet file from a local path or an s3:// URI."""
    if not path.startswith("s3://"):
        return pd.read_parquet(path)

    import boto3

    bucket, _, key = path[len("s3://"):].partition("/")
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _latest_local_snapshot(pattern: str) -> str:
    """Newest local snapshot matching a glob, or '' when nothing is cached.

    Snapshot keys are zero-padded, so lexicographic max is chronological max.
    """
    matches = glob.glob(pattern)
    return max(matches) if matches else ""


def load_news(path: str) -> pd.DataFrame:
    """Load the EOD news snapshot with parsed timestamps and mapped sentiment."""
    df = _read_parquet(path)

    df["ts"] = pd.to_datetime(
        df["date"], format="%a, %d %b %Y %H:%M:%S %z", utc=True, errors="coerce"
    )
    unparsed = int(df["ts"].isna().sum())
    if unparsed:
        print(f"WARNING: dropping {unparsed} news rows with unparsable dates")
        df = df.dropna(subset=["ts"])

    df["score"] = df["sentiment"].str.strip().str.lower().map(SENTIMENT_MAP)
    unmapped = int(df["score"].isna().sum())
    if unmapped:
        print(f"WARNING: {unmapped} news rows have unrecognised sentiment labels")

    df["is_core"] = ~df["source_name"].str.strip().str.lower().isin(TA_BOILERPLATE_SOURCES)
    topics_text = df["topics"].apply(
        lambda t: " ".join(str(x) for x in t) if t is not None else ""
    )
    df["event_text"] = (topics_text + " " + df["title"].fillna("")).str.lower()
    return df.sort_values("ts").reset_index(drop=True)


def load_bars(path: str) -> pd.DataFrame:
    """Load the D1 EOD price snapshot indexed by UTC bar timestamp."""
    df = _read_parquet(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    return df.sort_values("Timestamp").set_index("Timestamp")


def attribute_to_bars(news: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    """Map each news item to the index of the first bar closing at or after it.

    A D1 bar labelled day d closes at d+1 00:00 UTC. Items published after the
    final bar close have no decision point and are dropped.
    """
    close_times = (bars.index + pd.Timedelta(days=1)).values
    pos = np.searchsorted(close_times, news["ts"].values, side="left")
    return pd.Series(pos, index=news.index)


def build_panel(news: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Build the per-bar feature + forward-return panel."""
    pos = attribute_to_bars(news, bars)
    attributed = news.assign(bar_pos=pos)
    dropped = int((attributed["bar_pos"] >= len(bars)).sum())
    if dropped:
        print(f"NOTE: {dropped} news items published after the last bar close were dropped")
    attributed = attributed[attributed["bar_pos"] < len(bars)]

    grouped = attributed.groupby("bar_pos")
    core = attributed[attributed["is_core"]].groupby("bar_pos")

    panel = pd.DataFrame(index=range(len(bars)))
    panel["n_articles"] = grouped.size()
    panel["sent_mean"] = grouped["score"].mean()
    panel["sent_disp"] = grouped["score"].std()
    panel["n_articles_core"] = core.size()
    panel["sent_mean_core"] = core["score"].mean()

    for flag, pattern in TOPIC_FLAGS.items():
        hit = attributed["event_text"].str.contains(pattern, regex=True, na=False)
        panel[flag] = attributed.assign(hit=hit).groupby("bar_pos")["hit"].max()

    panel.index = bars.index
    panel["n_articles"] = panel["n_articles"].fillna(0)
    panel["n_articles_core"] = panel["n_articles_core"].fillna(0)
    for flag in TOPIC_FLAGS:
        panel[flag] = panel[flag].fillna(False).astype(float)

    panel["sent_mean_5d"] = panel["sent_mean"].rolling(5, min_periods=3).mean()
    panel["sent_mean_20d"] = panel["sent_mean"].rolling(20, min_periods=10).mean()
    rolling_n = panel["n_articles"].rolling(20, min_periods=10)
    panel["n_articles_z20"] = (panel["n_articles"] - rolling_n.mean()) / rolling_n.std()

    close = bars["Close"]
    panel["close"] = close
    for h in HORIZONS:
        panel[f"fwd_{h}"] = close.shift(-h) / close - 1.0

    covered = panel.index[panel["n_articles"] > 0]
    return panel.loc[covered.min():covered.max()].copy()


def rotation_null_p(feature: np.ndarray, target: np.ndarray, ic: float) -> float:
    """P-value from a circular-rotation null.

    Plain shuffling destroys autocorrelation and would understate the noise band
    for the rolling features; rotating preserves each series' own structure.
    """
    rng = np.random.default_rng(ROTATION_SEED)
    n = len(feature)
    offsets = rng.integers(1, n, size=N_ROTATIONS)
    extreme = 0
    for offset in offsets:
        rotated = np.concatenate([feature[offset:], feature[:offset]])
        null_ic = np.corrcoef(rotated, target)[0, 1]
        if abs(null_ic) >= abs(ic):
            extreme += 1
    return extreme / N_ROTATIONS


def bucket_returns(feature: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Mean forward return per feature bucket (quintiles, or 0/1 for flags)."""
    if feature.nunique() <= 2:
        groups = feature
    else:
        groups = pd.qcut(feature, 5, labels=False, duplicates="drop")
    table = target.groupby(groups).agg(["mean", "count"])
    table.index.name = "bucket"
    return table


def evaluate(panel: pd.DataFrame, feature: str, horizon: int) -> FeatureResult | None:
    """Compute IC, rotation p-value, half-sample ICs and buckets for one pair."""
    pair = panel[[feature, f"fwd_{horizon}"]].dropna()
    if len(pair) < 100 or pair[feature].nunique() < 2:
        return None

    x = pair[feature].to_numpy(dtype=float)
    y = pair[f"fwd_{horizon}"].to_numpy(dtype=float)
    ic = float(np.corrcoef(x, y)[0, 1])
    ic_spearman = float(stats.spearmanr(x, y).statistic)

    midpoint = len(pair) // 2
    halves = []
    for lo, hi in ((0, midpoint), (midpoint, len(pair))):
        xs, ys = x[lo:hi], y[lo:hi]
        halves.append(
            float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 and xs.std() > 0 else 0.0
        )

    buckets = bucket_returns(pair[feature], pair[f"fwd_{horizon}"])
    spread = float(buckets["mean"].iloc[-1] - buckets["mean"].iloc[0])

    return FeatureResult(
        feature=feature,
        horizon=horizon,
        n=len(pair),
        ic=ic,
        ic_spearman=ic_spearman,
        p_rotation=rotation_null_p(x, y, ic),
        noise_band=1.0 / np.sqrt(len(pair)),
        ic_first_half=halves[0],
        ic_second_half=halves[1],
        spread=spread,
        buckets=buckets,
    )


def _print_header(news: pd.DataFrame, bars: pd.DataFrame, panel: pd.DataFrame,
                  news_path: str, price_path: str) -> None:
    print("=" * 78)
    print("NEWS FORWARD-RETURN SIGNAL STUDY")
    print("=" * 78)
    print(f"news   : {news_path}")
    print(f"         {len(news):,} items  {news['ts'].min():%Y-%m-%d} -> {news['ts'].max():%Y-%m-%d}")
    print(f"price  : {price_path}")
    print(f"         {len(bars):,} D1 bars  {bars.index.min():%Y-%m-%d} -> {bars.index.max():%Y-%m-%d}")
    print(f"panel  : {len(panel):,} trading days  "
          f"{panel.index.min():%Y-%m-%d} -> {panel.index.max():%Y-%m-%d}")
    print(f"         {int((panel['n_articles'] == 0).sum())} days with no attributed news")


def _print_results(results: list[FeatureResult]) -> None:
    print()
    print("-" * 78)
    print("INFORMATION COEFFICIENTS")
    print("-" * 78)
    print(f"{'feature':<16}{'h':>3}{'n':>7}{'IC':>8}{'rho':>8}{'p':>8}"
          f"{'noise':>8}{'IC-1st':>9}{'IC-2nd':>9}  verdict")
    for r in results:
        verdict = "PASS" if r.passes else "-"
        print(f"{r.feature:<16}{r.horizon:>3}{r.n:>7}{r.ic:>8.3f}{r.ic_spearman:>8.3f}"
              f"{r.p_rotation:>8.3f}{r.noise_band:>8.3f}"
              f"{r.ic_first_half:>9.3f}{r.ic_second_half:>9.3f}  {verdict}")


def _print_buckets(results: list[FeatureResult], cost_rt: float, top: int = 4) -> None:
    print()
    print("-" * 78)
    print(f"BUCKET FORWARD RETURNS - {top} strongest |IC| (bp = basis points)")
    print("-" * 78)
    for r in sorted(results, key=lambda r: abs(r.ic), reverse=True)[:top]:
        print(f"\n{r.feature}  h={r.horizon}  IC={r.ic:+.3f}")
        for bucket, row in r.buckets.iterrows():
            print(f"    bucket {bucket:>4}: {row['mean'] * 1e4:>+8.2f} bp   n={int(row['count']):>5}")
        verdict = "below" if abs(r.spread) < cost_rt else "above"
        print(f"    top-bottom spread: {r.spread * 1e4:+.2f} bp  "
              f"({verdict} the {cost_rt * 1e4:.2f} bp round-trip cost)")


def _print_verdict(results: list[FeatureResult]) -> None:
    passed = [r for r in results if r.passes]
    expected_fp = ALPHA * len(results)
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"tests run                    : {len(results)}")
    print(f"expected false positives     : {expected_fp:.1f}  (alpha={ALPHA}, no correction)")
    print(f"features passing all gates   : {len(passed)}")
    print("gates: |IC| >= max(0.05, 2/sqrt(n)), rotation p < 0.05, "
          "and same IC sign in both halves")
    if not passed:
        print("\nNo feature clears the gates. On this panel the news snapshot shows no")
        print("forward-return signal worth wiring into the observation.")
    else:
        for r in passed:
            print(f"\n  {r.feature} h={r.horizon}: IC={r.ic:+.3f} p={r.p_rotation:.3f} "
                  f"halves={r.ic_first_half:+.3f}/{r.ic_second_half:+.3f}")
        print(f"\n{len(passed)} pass vs {expected_fp:.1f} expected by chance - "
              "treat as a lead to re-test out of sample, not a result.")


def run_study(news_path: str, price_path: str, spread_pips: float,
              out_dir: str | None) -> list[FeatureResult]:
    """Run the full study and print the report."""
    news = load_news(news_path)
    bars = load_bars(price_path)
    panel = build_panel(news, bars)
    _print_header(news, bars, panel, news_path, price_path)

    results = []
    for feature in FEATURES:
        for horizon in HORIZONS:
            result = evaluate(panel, feature, horizon)
            if result is not None:
                results.append(result)

    cost_rt = 2 * (spread_pips * PIP_SIZE_JPY) / float(panel["close"].median())
    _print_results(results)
    _print_buckets(results, cost_rt)
    _print_verdict(results)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        panel_path = os.path.join(out_dir, "news_panel.csv")
        panel.to_csv(panel_path)
        summary = pd.DataFrame([
            {
                "feature": r.feature, "horizon": r.horizon, "n": r.n, "ic": r.ic,
                "ic_spearman": r.ic_spearman, "p_rotation": r.p_rotation,
                "noise_band": r.noise_band, "ic_first_half": r.ic_first_half,
                "ic_second_half": r.ic_second_half, "spread": r.spread,
                "passes": r.passes,
            }
            for r in results
        ])
        summary_path = os.path.join(out_dir, "news_ic_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nWrote {panel_path}\nWrote {summary_path}")

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--news",
        default=_latest_local_snapshot(
            "s3data/*/marketdata/eod-news-snapshot/symbol=USD-JPY"
            "/year=*/month=*/day=*/*.parquet"
        ),
        help="EOD news snapshot parquet (local path or s3:// URI)",
    )
    parser.add_argument(
        "--price",
        default=_latest_local_snapshot(
            "s3data/*/marketdata/eod-snapshot/symbol=USDJPY/interval=D1"
            "/year=*/month=*/day=*/*.parquet"
        ),
        help="D1 EOD price snapshot parquet (local path or s3:// URI)",
    )
    parser.add_argument("--spread-pips", type=float, default=DEFAULT_SPREAD_PIPS,
                        help="One-way spread in pips, for the cost reference line")
    parser.add_argument("--out-dir", default=None,
                        help="Optional directory for the panel and summary CSVs")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.news or not args.price:
        raise SystemExit(
            "Could not resolve default snapshots under ./s3data - "
            "pass --news and --price explicitly."
        )
    run_study(args.news, args.price, args.spread_pips, args.out_dir)
