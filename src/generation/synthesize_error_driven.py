# src/generation/synthesize_error_driven.py
# Error-driven augmentation — generates genuine hedge variants from false positives.
#
# False positives are negatives the baseline classifier incorrectly flags as hedges.
# This pipeline inverts the contrastive direction: instead of generating non-hedges
# from hedge seeds, it generates genuine hedges from false positive seeds.
#
# Three classifier-specific seed files supported:
#   error_seeds_lr.parquet      -> error_driven_lr_raw.parquet
#   error_seeds_mlp1.parquet    -> error_driven_mlp1_raw.parquet
#   error_seeds_mlp2.parquet    -> error_driven_mlp2_raw.parquet
#
# Resume capability: skips already-completed seeds via checkpoint file.
# Generated variants are label=1 — genuine hedges produced from FP seeds.

import os
import json
import pandas as pd
import time
import random
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"

SECTORS = [
    "Technology", "Healthcare", "Financials", "Industrials",
    "Consumer Discretionary", "Energy", "Real Estate", "Materials"
]

def build_error_driven_prompt(seed_sentence: str, sector: str) -> str:
    return f"""You are correcting training data for a hedging language classifier.
You are working with earnings call transcripts in the {sector} sector.

The following sentence was incorrectly flagged as a hedge — it uses hedging-like
vocabulary but does NOT express genuine epistemic uncertainty about a future outcome:

FALSE POSITIVE: "{seed_sentence}"

Rewrite this sentence TWICE so that each version expresses GENUINE epistemic
uncertainty about a future outcome. Keep the same topic and vocabulary as closely
as possible — only add real uncertainty markers that reflect the speaker's genuine
doubt about the future.

A genuine hedge expresses the speaker's own forward-looking uncertainty.
Example of a FALSE POSITIVE: "We have a clear outlook on our revenue targets."
Example of a CORRECTED HEDGE: "We expect end market growth to be positive, though
the pace of recovery may be slower than we originally anticipated, subject to
macroeconomic conditions."
Do NOT use the vocabulary from the example above — it is only provided to
illustrate the structure of a genuine hedge.

Rules:
- Stay close to the seed's subject and vocabulary
- Express genuine forward-looking uncertainty — not past facts, not external events
- Variant 1 and Variant 2 MUST use different hedging devices — if Variant 1
  uses "may", Variant 2 should use "we believe", "we expect", "subject to", etc.
- Both variants must sound like plausible earnings call language
- Do not copy the seed — rewrite substantially
- Output only a JSON object, nothing else

Output format:
{{
  "false_positive": "{seed_sentence}",
  "hedge_variant_1": "first genuine hedge here",
  "hedge_variant_2": "second genuine hedge here"
}}"""


def generate_error_driven_variants(
    seed_sentence: str,
    sector: str,
    max_retries: int = 3,
) -> dict | None:
    prompt = build_error_driven_prompt(seed_sentence, sector)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400,
            )

            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)

            required_keys = {"false_positive", "hedge_variant_1", "hedge_variant_2"}
            if isinstance(result, dict) and required_keys.issubset(result.keys()):
                if all(isinstance(result[k], str) and len(result[k]) > 20
                       for k in ["hedge_variant_1", "hedge_variant_2"]):
                    return result

            print(f"Malformed output on attempt {attempt + 1}: {raw[:100]}")

        except json.JSONDecodeError:
            print(f"JSON parse error on attempt {attempt + 1}")
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")

        time.sleep(10)

    return None


def run_error_driven_generation(
    seeds_path: str,
    output_path: str,
    classifier_name: str,
    max_seeds: int = None,
) -> None:
    """
    Full generation loop with resume capability.
    Skips seeds already present in checkpoint file.

    Args:
        seeds_path:       Path to seed parquet (error_seeds_lr/mlp1/mlp2.parquet)
        output_path:      Where to save raw generated variants
        classifier_name:  'lr', 'mlp1', or 'mlp2' — for provenance metadata
        max_seeds:        Cap for dry runs
    """
    os.makedirs("data/synthetic", exist_ok=True)
    checkpoint_path = output_path.replace('.parquet', '_checkpoint.parquet')

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        existing     = pd.read_parquet(checkpoint_path)
        completed    = set(existing['seed_sentence'].unique())
        records      = existing.to_dict('records')
        print(f"Resuming from checkpoint: {len(existing)} variants already generated "
              f"({len(completed)} seeds completed).")
    else:
        completed = set()
        records   = []
        print("No checkpoint found — starting fresh.")

    seeds = pd.read_parquet(seeds_path)
    if max_seeds is not None:
        seeds = seeds.head(max_seeds)

    # Filter out already completed seeds
    seeds_remaining = seeds[
        ~seeds['sentence'].isin(completed)
    ].reset_index(drop=True)

    print(f"Total seeds: {len(seeds)} | "
          f"Completed: {len(completed)} | "
          f"Remaining: {len(seeds_remaining)}")

    failed = 0

    for i, row in seeds_remaining.iterrows():
        seed   = row['sentence']
        sector = row['sector']

        result = generate_error_driven_variants(
            seed_sentence=seed,
            sector=sector,
        )

        if result is None:
            failed += 1
            continue

        for variant_key in ["hedge_variant_1", "hedge_variant_2"]:
            records.append({
                "sentence":      result[variant_key],
                "label":         1,
                "source":        f"synthetic_error_driven_{classifier_name}",
                "seed_sentence": seed,
                "sector":        sector,
                "variant":       variant_key,
            })

        # Checkpoint every 50 seeds
        if (i + 1) % 50 == 0:
            df_checkpoint = pd.DataFrame(records)
            df_checkpoint.to_parquet(checkpoint_path, index=False)
            print(f"Processed {i + 1}/{len(seeds_remaining)} remaining | "
                  f"Generated: {len(records)} total | Failed: {failed}")

        # Rate limit buffer
        time.sleep(2)

    df_out = pd.DataFrame(records)
    df_out.to_parquet(output_path, index=False)
    print(f"\nDone. Generated {len(df_out)} hedge variants. Failed: {failed}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    CLASSIFIER_NAME='mlp2' #change to mlp1 or mlp2 for subsequent runs

    SEEDS_PATH   = f"data/synthetic/error_seeds_{CLASSIFIER_NAME}.parquet"
    OUTPUT_PATH  = f"data/synthetic/error_driven_{CLASSIFIER_NAME}_raw.parquet"
    # Start with Strategy B — diversity-based sampling
    run_error_driven_generation(
        seeds_path=SEEDS_PATH,
        output_path=OUTPUT_PATH,
        classifier_name=CLASSIFIER_NAME,
        max_seeds=None,
    )