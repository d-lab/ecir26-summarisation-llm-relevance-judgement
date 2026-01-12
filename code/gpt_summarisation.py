#! /usr/bin/env python
import argparse
import os
import sys
import json
from pathlib import Path
import time
import random

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import csv
from typing import Any, List

import numpy as np

#import bert_score
from openai import OpenAI


import re


tqdm.pandas()

# -------------------------------
# Data loading & prompt building
# -------------------------------

def _load_records(path):
    # supports either JSON lines (.jsonl) or a list in a .json
    p = Path(path)
    if p.suffix == ".jsonl":
        records = []
        for line in p.read_text(encoding="utf-8").splitlines():
            jl = json.loads(line)
            # jl['references'] = random.sample(jl['references'], 3)
            records.append(jl)
        return records
    else:
        return json.loads(p.read_text(encoding="utf-8"))
    



def load_docs():
    filename = os.getenv("DATA_FILENAME")
    if filename:
        if filename.endswith('.jsonl'):
            df =  pd.read_json(filename, lines=True)
            df.rename(columns={'text': 'doc_body'}, inplace=True)
            df = df.groupby('doc_body', as_index=False).first()
            df = df[['id', 'doc_body', 'references']]
            return df
        else:
            return pd.read_csv(filename)
    else:
        data_dir = os.getenv('DATA_DIR')
        dataset = os.getenv('DATASET_NAME')
        if not data_dir or not dataset:
            raise ValueError("DATA_DIR and DATASET_NAME must be set in environment.")
        path = Path(data_dir) / f"msmarco-passage-trec-{dataset}-judged/qrels/trec.qrel_docs.txt"

        return pd.read_csv(path)


def create_summarization_prompt(row: pd.Series, template: str) -> str:
    return template.replace("{DOC}", str(row.get('doc_body', '')))  # be defensive


# -------------------------------
# CSV writing helpers (newline modes)
# -------------------------------

def write_csv(df: pd.DataFrame, out_path: Path, summary_col: str, prompt_col: str, newline_mode: str) -> None:
    out_df = df.copy()
    # don't persist prompt column
    if prompt_col in out_df.columns:
        out_df.drop(columns=[prompt_col], inplace=True)

    if newline_mode == 'escape':
        out_df[summary_col] = (
            out_df[summary_col]
            .astype(str)
            .str.replace('\r\n', '\n')
            .str.replace('\n', r'\\n')
        )
        quoting = csv.QUOTE_MINIMAL
    elif newline_mode == 'space':
        out_df[summary_col] = (
            out_df[summary_col]
            .astype(str)
            .str.replace('\s+', ' ', regex=True)
            .str.strip()
        )
        quoting = csv.QUOTE_MINIMAL
    else:
        quoting = csv.QUOTE_ALL  # preserve real newlines with quoting

    out_df.to_csv(out_path, index=False, quoting=quoting, lineterminator='\n')


# -------------------------------
# Estimation helper
# -------------------------------

def estimate(text: str) -> int:
    """Return the word count for a single prompt string.

    Uses simple whitespace splitting as a rough proxy for token count.
    """
    if text is None:
        return 0
    return len(str(text).split())


# -------------------------------
# Responses API helpers (GPT-5)
# -------------------------------

def _extract_text_from_responses(results_jsonl: str, output_path: str = None) -> None:
    """Best-effort extraction of text from OpenAI Responses API result."""
    # 1) Direct convenience property
    records = _load_records(results_jsonl)
    summarisations_results = [r.get('output_text', '') for r in records]



def compute_geval(results_jsonl: str, output_path: str = None) -> None:
    """Compute GEVaL scores for summarization results."""
    records = _load_records(results_jsonl)
    with open ("./templates/g-eval_template.txt") as f:
        geval_template = f.read()

    client = build_client()
    responses = []
    for rec in tqdm(records, desc="Processing records for GEVaL"):
        prompt = geval_template.replace("{DOC}", str(rec.get('doc_body', ''))).replace("{SUMMARY}", str(rec.get('summarisation_result', '')))
        try:
            res = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(os.getenv("MAX_TOKENS", 300)),
                temperature=float(os.getenv("TEMPERATURE", 0.0)),
            )
            responses.append(res.choices[0].message.content.strip())
        except Exception as e:
            responses.append(f"ERROR: {e}")
        
        responses_dicts = []
        for resp in responses:
            resp = resp.replace("\n", "")
            match = re.search(r'\{.*?\}', resp, re.DOTALL)
            if match:
                resp = match.group(0)
            try:
                responses_dicts.append(json.loads(resp.replace("\n", "")))
            except Exception:
                responses_dicts.append({"response": resp})
        responses_df = pd.DataFrame(responses_dicts)
        responses_df.describe()
        if output_path:
            responses_df.to_csv(output_path, index=False)
    return responses


    cands = []
    refs_multi = []
    ids = []

    for rec in records:
        if not rec.get("summarisation_result"):  # skip if missing
            continue
        cands.append(rec["summarisation_result"])
        refs_multi.append(rec["references"])   # <- list of 11 refs
        ids.append(rec["id"])

    # Compute GEVaL scores (placeholder)
    gev_score = sum(len(c.split()) for c in cands) / sum(len(r.split()) for r in refs_multi) if refs_multi else 0

    # per-example scores
    per_item = [{
        "id": i,
        "GEVaL": float(gev_score)
    } for i in ids]

    # corpus-level means
    print("Corpus GEVaL:",
        "GEVaL=", float(np.mean([item["GEVaL"] for item in per_item])) if per_item else 0)
    gevaval_df = pd.DataFrame(per_item)
    if output_path:
        gevaval_df.to_csv(output_path, index=False)


# -------------------------------
# BERTScore (stub)
# -------------------------------
def compute_bertscore(results_jsonl: str, output_path: str = None) -> None:
    
    records = _load_records(results_jsonl)

    cands = []
    refs_multi = []
    ids = []

    for rec in records:
        if not rec.get("summarisation_result"):  # skip if missing
            continue
        cands.append(rec["summarisation_result"])
        refs_multi.append(rec["references"])   # <- list of 11 refs
        ids.append(rec["id"])

    P, R, F1 = bert_score.score(
        cands=cands,
        refs=refs_multi,
        lang="en",
        rescale_with_baseline=True    # recommended for English
    )

    # per-example scores
    per_item = [{
        "id": i,
        "P": float(p),
        "R": float(r),
        "F1": float(f)
    } for i, p, r, f in zip(ids, P.tolist(), R.tolist(), F1.tolist())]

    # corpus-level means
    print("Corpus BERTScore:",
        "P=", float(np.mean(P.numpy())),
        "R=", float(np.mean(R.numpy())),
        "F1=", float(np.mean(F1.numpy())))
    bertscore_df = pd.DataFrame(per_item)
    if output_path:
        bertscore_df.to_csv(output_path, index=False)

# -------------------------------
# GPT client factory
# -------------------------------

def build_client():
    if OpenAI is None:
        print("ERROR: openai package not installed. pip install openai>=1.0.0", file=sys.stderr)
        sys.exit(1)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)
    base_url = os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE')
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description='GPT Summarization (OpenAI)')
    parser.add_argument('--env-file', default='gpt.env', help='Path to env file with OPENAI_API_KEY (default: gpt.env)')
    parser.add_argument('--template-file', default='./templates/summarization_template.txt', help='Prompt template with {DOC} placeholder')
    parser.add_argument('--model', default='gpt-4o', help='OpenAI model name (e.g., gpt-5, gpt-4o, gpt-4o-mini)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (ignored for gpt-5 models)')
    parser.add_argument('--max-tokens', type=int, default=300, help='Max tokens for the summary (mapped to max_completion_tokens for gpt-5)')
    parser.add_argument('--batch-size', type=int, default=16, help='Number of rows per save window (requests are per-row)')
    parser.add_argument('--save-every', type=int, default=20, help='Save after this many batches')
    parser.add_argument('--output', default='gpt_summaries.csv', help='Output CSV with summary column')
    parser.add_argument('--resume', action='store_true', help='Resume from existing CSV; keeps already generated summaries')
    parser.add_argument('--run-subset', dest='run_subset', default=None,
                        help='Path to CSV file listing doc_id values to process; filters loaded docs to this subset')
    parser.add_argument('--newline-mode', choices=['keep','escape','space'], default='keep',
                        help="How to persist newlines in summaries: 'keep' (multiline quoted), 'escape' (\\n literals), 'space' (flatten)")
    parser.add_argument('--start-idx', type=int, default=0, help='Start row index (0-based, inclusive) to process')
    parser.add_argument('--end-idx', type=int, default=-1, help='End row index (0-based, exclusive). -1 means until end')
    parser.add_argument('--limit', type=int, default=0, help='If end-idx is -1, process at most this many rows from start-idx')
    parser.add_argument('--sleep', type=float, default=0.0, help='Sleep seconds between requests (basic rate limiting)')
    parser.add_argument('--max-retries', type=int, default=6, help='Max automatic retries on 429/5xx errors')
    parser.add_argument('--backoff-base', type=float, default=1.6, help='Exponential backoff base (seconds)')
    parser.add_argument('--backoff-cap', type=float, default=20.0, help='Maximum backoff delay (seconds)')
    parser.add_argument('--estimate', action='store_true', help='Only estimate total word count of prompts and exit')
    parser.add_argument('--debug', action='store_true', help='Print raw API responses when outputs are empty')
    # BERTScore options
    parser.add_argument('--compute-bertscore', action='store_true', help='Compute BERTScore from JSONL and exit')
    parser.add_argument('--compute-geval', action='store_true', help='Compute GEVaL from JSONL and exit')
    parser.add_argument('--summarisation-results-input', dest='summarisation_results_input', default=None,
                        help='Path to summarisation_results_input JSONL for BERTScore')
    args = parser.parse_args()

    # Load env
    load_dotenv(args.env_file)

    #GEval
    if args.compute_geval:
        if not args.summarisation_results_input or not str(args.summarisation_results_input).strip():
            print("ERROR: --summarisation-results-input JSONL path is required for --compute-geval", file=sys.stderr)
            sys.exit(2)
        inp_path = Path(args.summarisation_results_input)
        out_path = Path(args.output)
        if not inp_path.exists():
            print(f"ERROR: summarisation_results_input not found: {inp_path}", file=sys.stderr)
            sys.exit(2)
        compute_geval(str(inp_path), str(out_path))
        return


    # BERTScore mode (early exit)
    if args.compute_bertscore:
        if not args.summarisation_results_input or not str(args.summarisation_results_input).strip():
            print("ERROR: --summarisation-results-input JSONL path is required for --compute-bertscore", file=sys.stderr)
            sys.exit(2)
        inp_path = Path(args.summarisation_results_input)
        out_path = Path(args.output)
        if not inp_path.exists():
            print(f"ERROR: summarisation_results_input not found: {inp_path}", file=sys.stderr)
            sys.exit(2)
        compute_bertscore(str(inp_path), str(out_path))
        return

    # Data
    docs_df = load_docs()

    # Optional subset filtering by doc_id
    def _detect_id_col(df: pd.DataFrame) -> str:
        for c in ['doc_id', 'id', 'docid', 'document_id']:
            if c in df.columns:
                return c
        return None

    if args.run_subset:
        subset_path = Path(args.run_subset)
        if not subset_path.exists():
            print(f"ERROR: --run-subset file not found: {subset_path}", file=sys.stderr)
            sys.exit(2)
        try:
            subset_df = pd.read_csv(subset_path)
        except Exception:
            try:
                subset_df = pd.read_csv(subset_path, engine='python', escapechar='\\', on_bad_lines='warn')
            except Exception as e2:
                print(f"ERROR: failed to read --run-subset CSV: {e2}", file=sys.stderr)
                sys.exit(2)
        if 'doc_id' in subset_df.columns:
            subset_ids = subset_df['doc_id'].astype(str).dropna().unique().tolist()
        else:
            # fallback to first column
            subset_ids = subset_df.iloc[:, 0].astype(str).dropna().unique().tolist()
        id_col = _detect_id_col(docs_df)
        if not id_col:
            print("ERROR: Could not find an id column (e.g., doc_id or id) in loaded docs to apply subset filter.", file=sys.stderr)
            sys.exit(2)
        before_n = len(docs_df)
        docs_df = docs_df[docs_df[id_col].astype(str).isin(subset_ids)].reset_index(drop=True)
        after_n = len(docs_df)
        print(f"Applied subset filter from {subset_path.name}: kept {after_n} of {before_n} rows by {id_col}.")

    # Determine processing window
    total_rows = len(docs_df)
    start_window = max(0, args.start_idx)
    if args.end_idx != -1:
        end_window = min(args.end_idx, total_rows)
    else:
        end_window = min(start_window + args.limit, total_rows) if args.limit > 0 else total_rows

    if start_window >= end_window:
        print(f"No rows to process in range [{start_window}, {end_window}). Exiting.")
        return

    if start_window != 0 or end_window != total_rows:
        docs_df = docs_df.iloc[start_window:end_window]
        print(f"Processing window rows [{start_window}, {end_window}) -> {len(docs_df)} rows")

    # Prompt template and columns
    with open(args.template_file) as f:
        template = f.read()
    prompt_col = 'prompt_summary'
    if prompt_col not in docs_df.columns:
        docs_df[prompt_col] = docs_df.progress_apply(lambda row: create_summarization_prompt(row, template), axis=1)

    # If only estimation is requested, compute per-row word counts and exit.
    if args.estimate:
        docs_df['input_wordcount'] = docs_df[prompt_col].progress_apply(estimate)
        total_words = int(docs_df['input_wordcount'].sum())
        mean_wc = docs_df['input_wordcount'].mean()
        stdev_wc = docs_df['input_wordcount'].std()
        print(f"Mean prompt word count: {mean_wc:.2f}")
        print(f"Stddev prompt word count: {stdev_wc:.2f}")
        print(f"Total prompt words (windowed): {total_words}")
        print("Note: word count is a rough proxy for input tokens.")
        return

    summary_col = "summarisation_result"
    if summary_col not in docs_df.columns:
        docs_df[summary_col] = ''

    out_path = Path(args.output)

    # Resume
    start_pos = 0
    if args.resume and out_path.exists():
        try:
            existing = pd.read_csv(out_path)
        except Exception:
            # Fallback for messy CSVs
            try:
                existing = pd.read_csv(out_path, engine='python')
            except Exception as e:
                print(f"Warning: failed to read existing CSV for resume ({e}); skipping resume.")
                existing = None
        if existing is not None and summary_col in existing.columns:
            id_col_docs = _detect_id_col(docs_df)
            id_col_exist = _detect_id_col(existing)
            if args.run_subset and id_col_docs and id_col_exist:
                # Merge by id for subset-aware resume
                try:
                    existing_map = existing.set_index(id_col_exist)[summary_col]
                    def _pick_existing(r):
                        cur = r[summary_col]
                        if isinstance(cur, str) and len(cur) > 0:
                            return cur
                        key = str(r[id_col_docs])
                        val = existing_map.get(key)
                        if hasattr(val, 'iloc'):
                            try:
                                return val.iloc[0]
                            except Exception:
                                return ''
                        return val if isinstance(val, str) else ('' if pd.isna(val) else str(val))
                    docs_df[summary_col] = docs_df.apply(_pick_existing, axis=1)
                    mask_empty = docs_df[summary_col].isna() | (docs_df[summary_col].astype(str).str.len() == 0)
                    if (~mask_empty).all():
                        print('All records already summarized in this subset.')
                        write_csv(docs_df, out_path, summary_col, prompt_col, args.newline_mode)
                        return
                    first_empty_idx = mask_empty[mask_empty].index[0]
                    start_pos = docs_df.index.get_loc(first_empty_idx)
                    print(f'Resuming (subset) from window-relative index {start_pos}')
                except Exception as e:
                    print(f"Warning: subset-aware resume failed ({e}); falling back to positional resume.")
                    # fall through to positional resume
                    id_col_docs = None
            if not (args.run_subset and id_col_docs and id_col_exist):
                # align by position (original behavior)
                existing_slice = existing.iloc[0:len(docs_df)]
                if len(existing_slice) == len(docs_df):
                    docs_df[summary_col] = existing_slice[summary_col]
                    mask_empty = docs_df[summary_col].isna() | (docs_df[summary_col].astype(str).str.len() == 0)
                    if (~mask_empty).all():
                        print('All records already summarized in this window.')
                        write_csv(docs_df, out_path, summary_col, prompt_col, args.newline_mode)
                        return
                    # first empty position (relative)
                    first_empty_idx = mask_empty[mask_empty].index[0]
                    start_pos = docs_df.index.get_loc(first_empty_idx)
                    print(f'Resuming from window-relative index {start_pos}')
                else:
                    print('Warning: existing CSV length mismatch; ignoring positional resume.')
        else:
            print('Existing file lacks summary column; will overwrite on save.')

    # Build client
    client = build_client()

    # Generate
    batch_size = max(1, args.batch_size)
    save_every = max(1, args.save_every)

    def call_with_backoff(fn, *f_args, **f_kwargs):
        last_err = None
        for attempt in range(args.max_retries):
            try:
                return fn(*f_args, **f_kwargs)
            except Exception as e:
                last_err = e
                # Heuristics to detect retry-able errors (rate limit / transient)
                status = getattr(e, 'status_code', None) or getattr(e, 'status', None)
                msg = str(e).lower()
                retryable = (
                    status == 429 or
                    (isinstance(status, int) and status >= 500) or
                    'rate limit' in msg or 'temporarily unavailable' in msg or 'timeout' in msg
                )
                if attempt == args.max_retries - 1 or not retryable:
                    break
                # Exponential backoff with jitter
                delay = min(args.backoff_cap, (args.backoff_base ** attempt))
                delay *= 1 + random.random() * 0.25  # +0-25% jitter
                time.sleep(delay)
        # Final failure
        raise last_err

    def generate_summary(txt: str) -> str:
        # Call OpenAI API using the right interface per model.
        try:
            model_name = str(args.model)
            if model_name.startswith('gpt-5'):
                # Prefer Responses API for GPT-5
                def _call_responses():
                    return client.responses.create(
                        model=model_name,
                        input=txt,
                        max_output_tokens=args.max_tokens,
                    )
                res = call_with_backoff(_call_responses)
                content = _extract_text_from_responses(res)
                # Fallback: retry with structured input if empty
                if not content:
                    def _call_responses_struct():
                        return client.responses.create(
                            model=model_name,
                            input=[{
                                "role": "user",
                                "content": [{"type": "input_text", "text": txt}]
                            }],
                            max_output_tokens=args.max_tokens,
                        )
                    res2 = call_with_backoff(_call_responses_struct)
                    content2 = _extract_text_from_responses(res2)
                    if not content2 and args.debug:
                        try:
                            print(f"[DEBUG] GPT-5 raw response (structured): {res2}")
                        except Exception:
                            pass
                    content = content2
                if not content and args.debug:
                    try:
                        print(f"[DEBUG] GPT-5 raw response: {res}")
                    except Exception:
                        pass
                if not content and args.debug:
                    # Last-resort debugging fallback
                    return f"[NOTE] Empty GPT-5 output. Prompt echoed for debugging.\n{txt}"
                return (content or '').strip()
            else:
                # Chat Completions for non-GPT-5 models
                kwargs = {
                    'model': model_name,
                    'messages': [{"role": "user", "content": txt}],
                    'max_tokens': args.max_tokens,
                }
                kwargs['temperature'] = args.temperature
                res = call_with_backoff(
                    client.chat.completions.create,
                    **kwargs,
                )
                return (res.choices[0].message.content or '').strip()
        except Exception as e:
            print(f"OpenAI error (giving up): {e}", file=sys.stderr)
            return ''

    progress = tqdm(range(start_pos, len(docs_df)), desc='Summarizing')
    processed_since_save = 0
    for i in progress:
        if docs_df.at[docs_df.index[i], summary_col]:
            continue  # already filled (resume)
        prompt = docs_df.at[docs_df.index[i], prompt_col]
        summary = generate_summary(prompt)
        docs_df.at[docs_df.index[i], summary_col] = summary
        processed_since_save += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

        # Periodic save
        if processed_since_save >= batch_size or ((i + 1) % (batch_size * save_every) == 0):
            write_csv(docs_df, out_path, summary_col, prompt_col, args.newline_mode)
            docs_df.to_json(out_path.with_suffix('.jsonl'), orient='records', lines=True)
            processed_since_save = 0
            progress.set_postfix(saved=True)

    # Final save
    #write_csv(docs_df, out_path, summary_col, prompt_col, args.newline_mode)
    docs_df.to_json(out_path.with_suffix('.jsonl'), orient='records', lines=True)
    print('Done.')


if __name__ == '__main__':
    main()
