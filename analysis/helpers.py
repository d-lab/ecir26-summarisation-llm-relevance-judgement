# Agreement metrics (no fallbacks):
# - Cohen's κ (binary) via scikit-learn
# - Fleiss' κ (graded, 2 raters via ratings table) via statsmodels
# - Gwet AC1/AC2 implemented below (first-class, not fallback)
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score as sk_cohen_kappa
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa
from irrCAC.raw import CAC
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from pathlib import Path
import os, glob
from scipy.stats import kendalltau, spearmanr
from typing import Tuple


def read_qrels(qrels_path: str | Path) -> pd.DataFrame:
    """Read a TREC qrels file into a DataFrame with columns: query_id, doc_id, rel (int).
    Expected format per line: <query_id> <Q0|ignored> <doc_id> <rel>.
    Ignores blank lines and lines starting with '#'.
    """
    p = Path(qrels_path)
    if not p.exists():
        raise FileNotFoundError(f"qrels not found: {p}")
    # Robust whitespace-separated read; ignore comments
    df = pd.read_csv(
        p,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["query_id", "_q0", "doc_id", "rel"],
        dtype={"query_id": str, "doc_id": str},
    )
    # Drop any malformed rows
    # df = df.dropna(subset=["query_id", "doc_id", "rel"]).copy()
    # Ensure correct types
    df["query_id"] = df["query_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["rel"] = pd.to_numeric(df["rel"], errors="raise").astype(int)
    return df[["query_id", "doc_id", "rel"]]

def align_qrels(human_qrels: str | Path, llm_qrels: str | Path) -> pd.DataFrame:
    """Align human and LLM qrels on (query_id, doc_id) -> columns: query_id, doc_id, rel_h, rel_m."""
    df_h = read_qrels(human_qrels).rename(columns={"rel": "rel_h"})
    df_m = read_qrels(llm_qrels).rename(columns={"rel": "rel_m"})
    merged = (
        df_h.merge(df_m, on=["query_id", "doc_id"], how="inner", validate="one_to_one")
        .dropna(subset=["rel_h", "rel_m"])  # ensure both labels present
        .reset_index(drop=True)
    )
    if len(merged) == 0:
        raise ValueError("No overlapping (query_id, doc_id) pairs between human and LLM qrels.")
    return merged



def compute_agreement_metrics(human_qrels_path: str | Path,
                              llm_qrels_path: str | Path,
                              bin_threshold: int = 2) -> dict:
    """Compute agreement metrics between human and LLM qrels.
    Returns a dict with:
      - binary: cohen_kappa (sklearn), gwet_ac1
      - graded: fleiss_kappa (statsmodels), gwet_ac2
    """
    aligned = align_qrels(human_qrels_path, llm_qrels_path)
    aligned_rel = aligned[['rel_h','rel_m']]
    aligned_rel_binary = aligned_rel.apply(lambda x: x >= bin_threshold).astype(float)

    CAC_raters= CAC(aligned_rel_binary)

    cohen = sk_cohen_kappa(aligned_rel_binary['rel_h'],aligned_rel_binary['rel_m'])
    gwet_binary = CAC_raters.gwet()
    gwet_binary_score = gwet_binary['est']['coefficient_value']
    # print(f'gwet binary measure: {gwet_binary['est']['coefficient_name']}')


    CAC_raters= CAC(aligned_rel)
    fleiss = CAC_raters.fleiss()['est']['coefficient_value']
    gwet_graded = CAC_raters.gwet()
    gwet_graded_score = gwet_graded['est']['coefficient_value']
    # print(f'gwet graded measure: {gwet_graded['est']['coefficient_name']}')

    return {
        "n_items": int(len(aligned)),
        "binary_cohen_kappa": float(cohen), "binary_gwet": float(gwet_binary_score),
        "graded_fleiss_kappa": float(fleiss), "graded_gwet_graded": float(gwet_graded_score)}


def compute_system_ranking(
    qrels: Union[str, Path],
    runs_dir: Union[str, Path],
    ndcg_cutoff: int = 10,
    dataset_name: Optional[str] = None,
    scenario_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute per-system effectiveness and rankings using trectools given a QREL and a folder of TREC runs.

    Inputs:
      - qrels: path to a TREC qrels file (string or Path)
      - runs_dir: directory containing one TREC run file per system
      - ndcg_cutoff: cutoff k for NDCG@k
      - dataset_name, scenario_name: optional labels to annotate outputs

    Returns (three DataFrames):
      - df_scores: per-system scores with columns [system, map, ndcg@k, (dataset_name?), (scenario_name?)]
      - rank_map: df_scores + rank_map column, sorted by rank_map then system
      - rank_ndcg: df_scores + rank_ndcg column, sorted by rank_ndcg then system
    """
    # Import trectools lazily to avoid import errors at notebook import time
    try:
        from trectools import TrecQrel, TrecRun, TrecEval
    except Exception as e:
        raise RuntimeError("Missing dependency: trectools. Install via `pip install trectools`.") from e

    qrels_path = Path(qrels)
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels not found: {qrels_path}")

    run_paths = sorted(glob.glob(os.path.join(str(runs_dir), "*")))
    if not run_paths:
        raise FileNotFoundError(f"No run files found in: {runs_dir}")

    qrel = TrecQrel(str(qrels_path))

    rows = []
    for rp in run_paths:
        sys_name = os.path.basename(rp)
        try:
            run = TrecRun(rp)
            te = TrecEval(run, qrel)

            # MAP
            try:
                map_score = float(te.get_map())
            except Exception:
                map_score = float("nan")

            # NDCG@k (handle API variants across trectools versions)
            ndcg_val = None
            def _get_ndcg():
                # Preferred signatures
                for compute in (
                    lambda: te.get_ndcg(ndcg_cutoff),
                    lambda: te.get_ndcg(depth=ndcg_cutoff),
                ):
                    try:
                        return compute()
                    except Exception:
                        continue
                # Fallback to get_ndcg_cut_<k> if present
                meth = getattr(te, f"get_ndcg_cut_{ndcg_cutoff}", None)
                if callable(meth):
                    try:
                        return meth()
                    except Exception:
                        return None
                return None

            val = _get_ndcg()
            if isinstance(val, dict):
                # Mean of per-query values
                import numpy as _np
                ndcg_val = float(_np.mean(list(val.values())))
            elif val is not None:
                ndcg_val = float(val)
            else:
                ndcg_val = float("nan")

            row = {"system": sys_name, "map": map_score, f"ndcg@{ndcg_cutoff}": ndcg_val}
            if dataset_name is not None:
                row["dataset_name"] = str(dataset_name)
            if scenario_name is not None:
                row["scenario_name"] = str(scenario_name)
            rows.append(row)
        except Exception as e:
            print(f"[warn] Skipping {sys_name}: {e}")

    df_scores = pd.DataFrame(rows).sort_values("system").reset_index(drop=True)

    rank_map = (
        df_scores.assign(rank_map=df_scores["map"].rank(ascending=False, method="min"))
                 .sort_values(["rank_map", "system"])  # keep any extra columns
                 .reset_index(drop=True)
    )
    ndcg_col = f"ndcg@{ndcg_cutoff}"
    rank_ndcg = (
        df_scores.assign(rank_ndcg=df_scores[ndcg_col].rank(ascending=False, method="min"))
                 .sort_values(["rank_ndcg", "system"])
                 .reset_index(drop=True)
    )

    return df_scores, rank_map, rank_ndcg

# Simple correlation: single metric, same dataset, single model/scenario in llm_scores

def compute_correlation_simple(
    llm_scores: pd.DataFrame,
    human_scores: pd.DataFrame,
    metric: str,
    on: str = "system",
    include_rbo: bool = True,
    rbo_p: float = 0.9,
):
    """
    Compute Kendall's Tau and Pearson's r between llm_scores and human_scores for a single metric.
    Optionally compute Rank-Biased Overlap (RBO) between the induced rankings.

    Assumptions:
      - Both DataFrames are for the same dataset.
      - llm_scores corresponds to exactly one model and one scenario.
      - 'metric' exists as a column in both DataFrames.
      - Join key 'on' (default 'system') exists in both DataFrames.

    Returns:
      - If include_rbo is False (default): (kendall_tau, pearson_r, n_common, merged_df)
      - If include_rbo is True: (kendall_tau, pearson_r, rbo, n_common, merged_df)
        where merged_df has columns: [on, metric_human, metric_llm]
    """
    if on not in llm_scores.columns:
        if getattr(llm_scores.index, "name", None) == on:
            llm_scores = llm_scores.copy()
            llm_scores[on] = llm_scores.index
            llm_scores = llm_scores.reset_index(drop=True)
        else:
            raise ValueError(f"Join key '{on}' must exist in llm_scores DataFrame")

    if on not in human_scores.columns:
        if getattr(human_scores.index, "name", None) == on:
            human_scores = human_scores.copy()
            human_scores[on] = human_scores.index
            human_scores = human_scores.reset_index(drop=True)
        else:
            raise ValueError(f"Join key '{on}' must exist in human_scores DataFrame")

    # Validate metric presence
    if metric not in llm_scores.columns or metric not in human_scores.columns:
        raise ValueError(f"Metric '{metric}' must exist in both DataFrames")

    left = human_scores[[on, metric]].rename(columns={metric: f"{metric}_human"})
    right = llm_scores[[on, metric]].rename(columns={metric: f"{metric}_llm"})

    merged = pd.merge(left, right, on=on, how="inner")
    x = pd.to_numeric(merged[f"{metric}_human"], errors="coerce")
    y = pd.to_numeric(merged[f"{metric}_llm"], errors="coerce")
    mask = x.notna() & y.notna()
    n_common = int(mask.sum())

    if n_common < 2:
        if include_rbo:
            return float("nan"), float("nan"), float("nan"), n_common, merged[[on, f"{metric}_human", f"{metric}_llm"]]
        return float("nan"), float("nan"), n_common, merged[[on, f"{metric}_human", f"{metric}_llm"]]

    tau, _ = kendalltau(x[mask], y[mask])
    r, _ = spearmanr(x[mask], y[mask])

    if not include_rbo:
        return float(tau), float(r), n_common, merged[[on, f"{metric}_human", f"{metric}_llm"]]

    # Compute RBO between rankings induced by the metric (descending). Break ties by 'on' for determinism.
    try:
        aligned = merged.loc[mask, [on, f"{metric}_human", f"{metric}_llm"]].copy()
        list_h = aligned.sort_values([f"{metric}_human", on], ascending=[False, True])[on].astype(str).tolist()
        list_m = aligned.sort_values([f"{metric}_llm", on], ascending=[False, True])[on].astype(str).tolist()
        rbo_score = float("nan")
        try:
            import rbo as _rbo
            if hasattr(_rbo, "RankingSimilarity"):
                rbo_score = float(_rbo.RankingSimilarity(list_h, list_m).rbo(p=rbo_p))
            elif hasattr(_rbo, "RBO"):
                rbo_score = float(_rbo.RBO(list_h, list_m, p=rbo_p).rbo())
            else:
                # Some variants expose a function 'rbo' taking lists and p
                func = getattr(_rbo, "rbo", None)
                if callable(func):
                    rbo_score = float(func(list_h, list_m, p=rbo_p))
        except Exception as e:
            print(f"[warn] RBO computation failed: {e}")
            rbo_score = float("nan")
    except Exception as e:
        print(f"[warn] RBO preparation failed: {e}")
        rbo_score = float("nan")

    return float(tau), float(r), rbo_score, n_common, merged[[on, f"{metric}_human", f"{metric}_llm"]]
