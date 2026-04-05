# src/generation/synthesize_hard_contrastive.py
# Hard contrastive negative generation for hedging language classification.
#
# Generates Type 3 hard negatives only — sentences that use hedging surface
# markers (may, might, we believe, could) but where the uncertainty is NOT
# the speaker's epistemic stance. Two subtypes:
#   - Epistemic verb reporting past fact: "We believe margins expanded last quarter"
#   - Modal describing external possibility: "Regulators could approve this by year end"
#
# Two Type 3 negatives generated per seed sentence.
# These are the hardest negatives — surface-identical to hedges but semantically
# non-epistemic. Requires more nuanced generation than Type 1/2.
# Begin with 8B model for consistency with other experiments.
# Repeat with 70B model in a separate run for quality comparison.

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
def build_hard_contrastive_prompt(seed_sentence: str, sector: str) -> str:
    """
    Builds a Type 3 hard contrastive negative generation prompt.
    Generates two sentences that use hedging surface markers but are
    not epistemically hedged — the hardest negatives to distinguish
    from true hedges.

    Two subtypes, one each:
    - Subtype A: epistemic verb used to report a past fact
    - Subtype B: modal verb describing external possibility, not speaker uncertainty

    Args:
        seed_sentence: Real positive example from labeled dataset
        sector: GICS sector of source company

    Returns:
        Prompt string
    """
    return f"""You are generating training data for a hedging language classifier.
You are working with earnings call transcripts in the {sector} sector.

A hedged statement expresses the SPEAKER'S uncertainty about a future claim.
Example of genuine hedge: "We expect margins may decline next quarter."

A hard negative looks like a hedge on the surface but is NOT epistemically hedged.
There are two types of hard negatives:

Subtype A — Epistemic verb reporting past fact:
The verb (believe, think) is present but describes something already known, not uncertain.
Example: "We believe margins expanded last quarter." — no uncertainty, reporting known fact.

Subtype B — Modal describing external possibility:
The modal (may, might, could) is present but describes an external event, not speaker uncertainty.
Example: "Regulators could approve the merger by year end." — speaker has no uncertainty about this claim.

SEED: "{seed_sentence}"

Generate exactly two hard negatives based on this seed:
- Negative 1: Subtype A — use an epistemic verb (believe, think, expect) to report a past fact related to the seed's topic
- Negative 2: Subtype B — use a modal (may, might, could) to describe an external possibility related to the seed's topic

Rules:
- Stay close to the seed's subject and vocabulary
- Do NOT express the speaker's own uncertainty about future outcomes
- Both must sound like plausible earnings call language
- Output only a JSON object, nothing else

Output format:
{{
  "positive": "{seed_sentence}",
  "subtype_a": "epistemic verb reporting past fact here",
  "subtype_b": "modal describing external possibility here"
}}"""

def generate_hard_contrastive_variants(
    seed_sentence: str,
    sector: str,
    max_retries: int = 3,
) -> dict | None:
    """
    Calls Groq API to generate Type 3 hard contrastive negatives.
    
    Args:
        seed_sentence: A real positive example from the labeled dataset
        sector: GICS sector of the source company
        max_retries: Number of retries on API failure or malformed output
    
    Returns:
        Dict with keys: positive, subtype_a, subtype_b.
        Returns None on failure.
    
    Notes:
        - subtype_a: epistemic verb reporting past fact
        - subtype_b: modal describing external possibility
        - Both are label=0 — not hedges despite surface similarity
    """
    prompt = build_hard_contrastive_prompt(seed_sentence, sector)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=500,
            )

            raw = response.choices[0].message.content.strip()

            # Parse JSON output
            result = json.loads(raw)

            # Validate output structure
            required_keys = {"positive", "subtype_a", "subtype_b"}
            if isinstance(result, dict) and required_keys.issubset(result.keys()):
                if all(isinstance(result[k], str) and len(result[k]) > 20
                       for k in ["subtype_a", "subtype_b"]):
                    return result

            print(f"Malformed output on attempt {attempt + 1}: {raw[:100]}")

        except json.JSONDecodeError:
            print(f"JSON parse error on attempt {attempt + 1}")
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")

        time.sleep(10)

    return None
def run_hard_contrastive_generation(
    positives_path: str = "data/processed/train.parquet",
    output_path: str = "data/synthetic/hard_contrastive_raw.parquet",
    max_seeds: int = None,
) -> None:
    """
    Full generation loop for Type 3 hard contrastive negative augmentation.
    Generates two hard negatives per seed — one subtype A, one subtype B.
    
    Args:
        positives_path: Path to real positive examples
        output_path: Where to save raw hard contrastive outputs
        max_seeds: If set, limits generation for dry runs
    
    Notes:
        - Subtype A: epistemic verb reporting past fact
        - Subtype B: modal describing external possibility
        - Both labeled 0 — not hedges despite surface similarity
        - Checkpoint saved every 100 seeds
    """
    os.makedirs("data/synthetic", exist_ok=True)

    positives = pd.read_parquet(positives_path)
    if 'label' in positives.columns:
        positives = positives[positives['label'] == 1].reset_index(drop=True)
    if max_seeds is not None:
        positives = positives.head(max_seeds)
    print(f"Loaded {len(positives)} seed positives.")

    records = []
    failed = 0

    for i, row in positives.iterrows():
        seed = row['sentence']
        sector = row['sector']

        result = generate_hard_contrastive_variants(
            seed_sentence=seed,
            sector=sector,
        )

        if result is None:
            failed += 1
            continue

        # Subtype A record
        records.append({
            "sentence": result['subtype_a'],
            "label": 0,
            "source": "synthetic_hard_contrastive_a",
            "seed_sentence": seed,
            "sector": sector,
            "contrastive_type": "subtype_a",
        })

        # Subtype B record
        records.append({
            "sentence": result['subtype_b'],
            "label": 0,
            "source": "synthetic_hard_contrastive_b",
            "seed_sentence": seed,
            "sector": sector,
            "contrastive_type": "subtype_b",
        })

        # Checkpoint every 100 seeds
        if (i + 1) % 100 == 0:
            df_checkpoint = pd.DataFrame(records)
            df_checkpoint.to_parquet(
                output_path.replace('.parquet', '_checkpoint.parquet'),
                index=False
            )
            print(f"Processed {i + 1}/{len(positives)} seeds | "
                  f"Generated: {len(records)} | Failed: {failed}")

    df_out = pd.DataFrame(records)
    df_out.to_parquet(output_path, index=False)
    print(f"\nDone. Generated {len(df_out)} hard contrastive negatives. Failed: {failed}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    run_hard_contrastive_generation(
        positives_path="data/processed/train.parquet",
        output_path="data/synthetic/hard_contrastive_raw.parquet",
        max_seeds=None,
    )