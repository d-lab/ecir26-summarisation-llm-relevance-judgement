#! /usr/bin/env python
"""
CLI: Run GPT relevance judgments over an inputs-like file.

Overview
- Reads a JSONL/CSV of rows containing (query_id, doc_id, query) and a passage text.
- Picks passage from text_summary (preferred) or falls back to text/doc_body/passage.
- Builds a prompt from templates/relevance_judgment_template.txt with {query} and {passage}.
- Calls OpenAI (default gpt-4o) to get a relevance score (0–3) and parses it.
- Writes results to CSV and JSONL, with resume support, batching, and retry backoff.

Key columns added
- passage_source: which column supplied the passage (text_summary/text/doc_body/passage).
- passage_used: the actual passage text sent to the model.
- prompt_relevance: the final prompt string sent.
- model_output: raw model response text.
- rel_score: parsed integer score in [0,3], or -1 if not parsed.
"""
import argparse
import os
import sys
import json
from pathlib import Path
import time
import random
import re

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI


#Input: JSONL/CSV with columns: query_id, doc_id, query, and one of:
#Preferred: text_summary
#Fallbacks: text, doc_body, or passage
#Prompting: Loads relevance_judgment_template.txt, substitutes {query} and {passage}.
#Model: GPT-4o by default (configurable), temperature 0.0, max_tokens 64 (configurable).
#Output: CSV (default summarisation_outputs/gpt_relevance_judgements.csv) and a matching JSONL.
#Resume: If --resume and output exists, rows with existing rel_score are skipped.
#Robustness: Retries on rate-limit/transient errors with jittered backoff; saves every N rows.


def build_client() -> OpenAI:
    """Create and return an OpenAI client using env vars.

    Honors OPENAI_API_KEY (required) and OPENAI_BASE_URL/OPENAI_API_BASE (optional).
    Exits with an error if the API key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def load_inputs(path: Path) -> pd.DataFrame:
    """Load inputs from JSONL or CSV into a DataFrame and normalize columns.

    Ensures presence of common columns (query_id, doc_id, query, text_summary, text,
    doc_body, passage) with None defaults when missing and standardizes string types
    for identifiers.
    """
    if not path.exists():
        print(f"ERROR: input file not found: {path}", file=sys.stderr)
        sys.exit(2)
    if path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    # normalize common columns
    # prefer text_summary as the passage, fallback to text/doc_body
    for col in ["query_id", "doc_id", "query", "text_summary", "text", "doc_body", "passage"]:
        if col not in df.columns:
            df[col] = None
    # Ensure string types where applicable
    for c in ["query_id", "doc_id", "query"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def choose_passage(row: pd.Series) -> tuple[str, str]:
    """Choose the best available passage for a row.

    Preference order: text_summary > text > doc_body > passage.
    Returns (source_col, text). If none is present, returns ("", "").
    Skips strings that are empty or the literal "nan".
    """
    # returns (source_col, text)
    val = row.get("text_summary", None)
    if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "nan":
        return ("text_summary", str(val))
    for col in ["text", "doc_body", "passage"]:
        v = row.get(col, None)
        if pd.notna(v) and str(v).strip() and str(v).strip().lower() != "nan":
            return (col, str(v))
    return ("", "")


def build_prompt(template: str, query: str, passage: str) -> str:
    """Render the relevance prompt by substituting {query} and {passage}."""
    # Use simple replace to avoid issues with braces in templates
    return template.replace("{query}", str(query)).replace("{passage}", str(passage))


def extract_score(text: str) -> int:
    """Extract the integer relevance score from the model output.

    Primary pattern: 'final score: X' (case-insensitive). Fallback: first standalone
    digit 0–3 in the response. Returns -1 if no score is detected.
    """
    if not text:
        return -1
    m = re.search(r"final\s*score\s*:\s*([0-3])", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # fallback: any standalone 0-3
    m2 = re.search(r"\b([0-3])\b", text)
    if m2:
        return int(m2.group(1))
    return -1


def main():
    # 1) Parse CLI arguments (model, I/O paths, batching, retries, etc.).
    parser = argparse.ArgumentParser(description="GPT relevance judgement CLI")
    parser.add_argument("--env-file", default="./env_files/gpt_dl2019.env", help="Path to .env with OPENAI_API_KEY")
    parser.add_argument("--input", required=True, help="Path to inputs JSONL/CSV (e.g., inputs_from_summaries.jsonl)")
    parser.add_argument("--template-file", default="./templates/relevance_judgment_template.txt", help="Template with {query} and {passage}")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--jsonl-output", default=None, help="JSONL output path (defaults to output with .jsonl)")
    parser.add_argument("--qrels-output", default="./summarisation_outputs/gpt_4o_summary_umbrella_zeroshot_qrels.txt", help="Optional TREC qrels txt path (topic q0 docid rel)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output (skip rows with scores)")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many rows from start")
    parser.add_argument("--batch-size", type=int, default=16, help="Save every N processed rows")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between calls")
    parser.add_argument("--max-retries", type=int, default=6, help="Max retries for transient errors")
    parser.add_argument("--backoff-base", type=float, default=1.6, help="Exponential backoff base")
    parser.add_argument("--backoff-cap", type=float, default=20.0, help="Exponential backoff cap")
    args = parser.parse_args()

    # 2) Load environment variables (API key, base URL).
    load_dotenv(args.env_file)

    # 3) Load input rows (JSONL/CSV) and normalize columns.
    inp_path = Path(args.input)
    df = load_inputs(inp_path)

    # 4) Determine processing window (start/limit) for large files.
    start = max(0, args.start_idx)
    end = len(df) if args.limit <= 0 else min(len(df), start + args.limit)
    if start >= end:
        print(f"No rows to process in range [{start}, {end}). Exiting.")
        return
    df = df.iloc[start:end].copy()


    jsonl_path = Path(args.jsonl_output) if args.jsonl_output else Path("./summarisation_outputs/gpt_relevance_judgements.jsonl")
    qrels_path = Path(args.qrels_output) if args.qrels_output else None

    # 6) Load the relevance judgment template (expects {query} and {passage}).
    with open(args.template_file, "r", encoding="utf-8") as f:
        template = f.read()

    # 7) Build resume state and merge prior results, if any.
    #    Keys used: (query_id, doc_id) when present.
    done_keys = set()
    prev = None
    if args.resume and jsonl_path.exists():
        try:
            prev = pd.read_json(jsonl_path, lines=True)
        except Exception:
            prev = None
    if prev is not None and len(prev) > 0:
        try:
            key_cols_prev = [c for c in ["query_id", "doc_id"] if c in prev.columns]
            key_cols_df = [c for c in ["query_id", "doc_id"] if c in df.columns]
            key_cols = [c for c in ["query_id", "doc_id"] if c in key_cols_prev and c in key_cols_df]
            # Track completed keys (rows with a non-null rel_score)
            if key_cols and "rel_score" in prev.columns:
                for _, r in prev.dropna(subset=["rel_score"]).iterrows():
                    key = tuple(str(r.get(c, "")) for c in key_cols)
                    done_keys.add(key)

            # Merge previous outputs into current df to preserve values on resume
            out_cols = [
                "passage_source",
                "passage_used",
                "prompt_relevance",
                "model_output",
                "rel_score",
            ]
            if key_cols:
                prev_subset_cols = [c for c in key_cols + out_cols if c in prev.columns]
                if all(k in prev_subset_cols for k in key_cols):
                    prev_subset = prev[prev_subset_cols].copy()
                    # Drop duplicate keys keeping the last occurrence
                    prev_subset = prev_subset.drop_duplicates(subset=key_cols, keep="last")
                    merged = df.merge(prev_subset, on=key_cols, how="left", suffixes=("", "_prev"))
                    for col in out_cols:
                        col_prev = f"{col}_prev"
                        if col in merged.columns and col_prev in merged.columns:
                            merged[col] = merged[col].combine_first(merged[col_prev])
                    # Drop *_prev helper columns
                    drop_cols = [c for c in merged.columns if c.endswith("_prev")]
                    if drop_cols:
                        merged.drop(columns=drop_cols, inplace=True)
                    df = merged
        except Exception:
            # Non-fatal; continue without merge if anything goes wrong
            pass

    # 8) Create the OpenAI client for API calls.
    client = build_client()

    # 9) Ensure result columns exist but do not overwrite existing values (from resume merge).
    for _col in ["passage_source", "passage_used", "prompt_relevance", "model_output", "rel_score"]:
        if _col not in df.columns:
            df[_col] = None

    # 10) Define a helper for retries with exponential backoff and jitter
    #     for transient errors such as rate-limits and 5xx responses.
    def call_with_backoff(callable_fn, **kwargs):
        last_err = None
        for attempt in range(args.max_retries):
            try:
                return callable_fn(**kwargs)
            except Exception as e:
                last_err = e
                status = getattr(e, "status_code", None) or getattr(e, "status", None)
                msg = str(e).lower()
                retryable = (status == 429) or (isinstance(status, int) and status >= 500) or ("rate limit" in msg) or ("timeout" in msg) or ("temporarily" in msg)
                if attempt == args.max_retries - 1 or not retryable:
                    break
                delay = min(args.backoff_cap, (args.backoff_base ** attempt))
                delay *= 1 + random.random() * 0.25
                time.sleep(delay)
        raise last_err

    # Helper to write TREC qrels file (topic q0 docid rel)
    def write_qrels_file(_df: pd.DataFrame, _path: Path):
        if _path is None:
            return
        if "query_id" in _df.columns and "doc_id" in _df.columns and "rel_score" in _df.columns:
            proc_df = _df[_df["rel_score"].notna()].copy()
            with open(_path, "w", encoding="utf-8") as f:
                for r in proc_df.itertuples(index=False):
                    qid = getattr(r, "query_id", None)
                    did = getattr(r, "doc_id", None)
                    rel = getattr(r, "rel_score", None)
                    if qid is None or did is None:
                        continue
                    try:
                        rel_int = int(rel) if rel is not None and str(rel).strip() != "" else 0
                    except Exception:
                        rel_int = 0
                    f.write(f"{qid} q0 {did} {rel_int}\n")

    # Helper to append only new qrels lines (no rewrite)
    def append_qrels_file(_df: pd.DataFrame, _path: Path):
        if _path is None:
            return
        if "query_id" in _df.columns and "doc_id" in _df.columns and "rel_score" in _df.columns:
            with open(_path, "a", encoding="utf-8") as f:
                for r in _df.itertuples(index=False):
                    qid = getattr(r, "query_id", None)
                    did = getattr(r, "doc_id", None)
                    rel = getattr(r, "rel_score", None)
                    if qid is None or did is None or rel is None:
                        continue
                    try:
                        rel_int = int(rel) if str(rel).strip() != "" else 0
                    except Exception:
                        rel_int = 0
                    f.write(f"{qid} q0 {did} {rel_int}\n")

    # Helper: order rows by their order in the original input
    def sort_by_input_order(_df: pd.DataFrame, _order_map: dict | None, _key_cols: list[str]):
        if not _key_cols or not isinstance(_order_map, dict) or len(_order_map) == 0:
            return _df
        tmp = _df.copy()
        try:
            tmp["__key_tuple__"] = tmp[_key_cols].astype(str).agg(tuple, axis=1)
            tmp["__order__"] = tmp["__key_tuple__"].map(_order_map)
            tmp["__order__"] = tmp["__order__"].fillna(float("inf"))
            tmp = tmp.sort_values(by="__order__", kind="stable")
            tmp = tmp.drop(columns=["__key_tuple__", "__order__"])
            return tmp
        except Exception:
            return _df

    # 11) Iterate rows: build prompt, call model, parse score, persist, save in batches.
    processed = 0
    batch_since_save = 0
    rows = list(df.itertuples(index=False))
    pbar = tqdm(total=len(rows), desc="Relevance judging")
    key_cols = [c for c in ["query_id", "doc_id"] if c in df.columns]
    # Build order map from the full input to preserve final ordering
    order_map = None
    try:
        if key_cols:
            base_df = load_inputs(inp_path)
            base_df = base_df.copy()
            base_df["__key_tuple__"] = base_df[key_cols].astype(str).agg(tuple, axis=1)
            order_map = {k: i for i, k in enumerate(base_df["__key_tuple__"]) }
    except Exception:
        order_map = None

    for idx, row in enumerate(rows):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row._mapping) if hasattr(row, "_mapping") else dict(row._asdict())
        # Skip if already done (resume)
        if key_cols:
            key = tuple(str(row_dict.get(c, "")) for c in key_cols)
            if key in done_keys:
                pbar.update(1)
                continue

        query = row_dict.get("query", "")
        src_col, passage = choose_passage(pd.Series(row_dict))
        prompt = build_prompt(template, query=query, passage=passage)

        # Call model
        try:
            res = call_with_backoff(
                client.chat.completions.create,
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            out_txt = (res.choices[0].message.content or "").strip()
        except Exception as e:
            out_txt = f"ERROR: {e}"

        score = extract_score(out_txt)

        # Persist into df
        df.at[df.index[idx], "passage_source"] = src_col
        df.at[df.index[idx], "passage_used"] = passage
        df.at[df.index[idx], "prompt_relevance"] = prompt
        df.at[df.index[idx], "model_output"] = out_txt
        df.at[df.index[idx], "rel_score"] = score

        processed += 1
        batch_since_save += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

        # Save batch
        if batch_since_save >= max(1, args.batch_size):
            # Persist only processed rows in this window
            processed_df = df[df["rel_score"].notna()].copy()
            if args.resume and jsonl_path.exists():
                # Append only new rows (not in done_keys)
                if key_cols:
                    processed_df["__key_tuple__"] = processed_df[key_cols].astype(str).agg(tuple, axis=1)
                    to_write = processed_df[~processed_df["__key_tuple__"].isin(done_keys)].copy()
                    # Deduplicate new rows on key to avoid double writes within the same batch
                    to_write = to_write.drop_duplicates(subset=key_cols, keep="last")
                    to_write.drop(columns=["__key_tuple__"], inplace=True)
                else:
                    to_write = processed_df
                # Sort new rows by input order before appending
                to_write = sort_by_input_order(to_write, order_map, key_cols)
                if not to_write.empty:
                    json_ready = to_write.where(pd.notna(to_write), None)
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        for rec in json_ready.to_dict(orient="records"):
                            f.write(json.dumps(rec, ensure_ascii=False))
                            f.write("\n")
                    # Append qrels for new rows
                    append_qrels_file(to_write, qrels_path)
                    # Update done_keys to avoid duplicate appends in this run
                    if key_cols:
                        for k in to_write[key_cols].astype(str).agg(tuple, axis=1).tolist():
                            done_keys.add(k)
            else:
                # Not resuming: rewrite files with processed rows of the current window
                processed_df = sort_by_input_order(processed_df, order_map, key_cols)
                save_df = processed_df.where(pd.notna(processed_df), None)
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for rec in save_df.to_dict(orient="records"):
                        f.write(json.dumps(rec, ensure_ascii=False))
                        f.write("\n")
                write_qrels_file(save_df, qrels_path)
            batch_since_save = 0
        pbar.update(1)

    pbar.close()
    # 12) Final save to CSV and JSONL.
    # Final save: write only incremental new rows when resuming; otherwise rewrite window
    processed_df = df[df["rel_score"].notna()].copy()
    if args.resume and jsonl_path.exists():
        if key_cols:
            processed_df["__key_tuple__"] = processed_df[key_cols].astype(str).agg(tuple, axis=1)
            to_write = processed_df[~processed_df["__key_tuple__"].isin(done_keys)].copy()
            # Deduplicate on keys to avoid duplicate final writes
            to_write = to_write.drop_duplicates(subset=key_cols, keep="last")
            to_write.drop(columns=["__key_tuple__"], inplace=True)
        else:
            to_write = processed_df
        # Sort by original input order
        to_write = sort_by_input_order(to_write, order_map, key_cols)
        if not to_write.empty:
            json_ready = to_write.where(pd.notna(to_write), None)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                for rec in json_ready.to_dict(orient="records"):
                    f.write(json.dumps(rec, ensure_ascii=False))
                    f.write("\n")
            append_qrels_file(to_write, qrels_path)
        # Final safeguard: dedupe the whole JSONL by (query_id, doc_id) and regenerate qrels
        try:
            if key_cols:
                all_df = pd.read_json(jsonl_path, lines=True)
                all_df = all_df.drop_duplicates(subset=key_cols, keep="last")
                # Keep only processed rows
                all_df = all_df[all_df["rel_score"].notna()].copy()
                # Sort full set by input order
                all_df = sort_by_input_order(all_df, order_map, key_cols)
                # Rewrite JSONL with deduped records
                json_ready = all_df.where(pd.notna(all_df), None)
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for rec in json_ready.to_dict(orient="records"):
                        f.write(json.dumps(rec, ensure_ascii=False))
                        f.write("\n")
                # Rewrite qrels to match deduped set
                write_qrels_file(all_df, qrels_path)
        except Exception:
            pass
    else:
        processed_df = sort_by_input_order(processed_df, order_map, key_cols)
        save_df = processed_df.where(pd.notna(processed_df), None)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in save_df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
        write_qrels_file(save_df, qrels_path)

    print(f"Done. Wrote {jsonl_path}{' and ' + str(qrels_path) if qrels_path else ''}")


if __name__ == "__main__":
    main()
