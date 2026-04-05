# src/generation/synthesize_contrastive.py
# Contrastive hard negative generation for hedging language classification.
#
# Generates three types of hard negatives from real positive seed sentences:
#   Type 1 — Factual rewrite with negated claim (alternating by seed index)
#   Type 2 — Factual rewrite asserting same claim without uncertainty (alternating)
#   Type 3 — Hedge-sounding but non-epistemic (always generated)
#
# Type 3 uses two subtypes:
#   - Epistemic verb reporting past fact: "We believe margins expanded last quarter"
#   - Modal describing external possibility: "Regulators could approve this by year end"
#
# Filtering direction is reversed relative to positive augmentation:
# retain only synthetics with LOW calibrated score and NARROW interval width.

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
def build_contrastive_prompt(seed_sentence: str, sector: str) -> str:
    """
    Builds a contrastive hard negative generation prompt.
    Generates one Type 1 and one Type 2 negative per seed.
    
    Type 1 — Factual rewrite with negated claim
    Type 2 — Factual rewrite asserting same claim without uncertainty
    
    Args:
        seed_sentence: Real positive example from labeled dataset
        sector: GICS sector of source company
    
    Returns:
        Prompt string
    """
    return f"""You are generating training data for a hedging language classifier.
You are working with earnings call transcripts in the {sector} sector.

SEED: "{seed_sentence}"

Generate exactly two hard negatives:

Type 1 — Factual rewrite, negated claim:
Rewrite the seed as a confident factual statement where the claim is NEGATED.
No uncertainty markers. Past tense preferred.
Stay close to the seed's subject — do not substitute unrelated financial terms.
Example: "We expect ocean rates may remain elevated" → "Ocean rates normalized in Q3."

Type 2 — Factual rewrite, same claim:
Rewrite the seed as a confident factual assertion of the SAME claim.
Remove ALL hedging language — no may, might, could, believe, expect, think,
subject to, uncertain, or any epistemic markers. State it as established fact.
Stay close to the seed's subject.
Example: "We expect ocean rates may remain elevated" → "Ocean rates remained elevated throughout the year."

Rules:
- Stay close to the seed's subject and vocabulary
- Do not copy the seed — rewrite substantially
- Both must sound like plausible earnings call language
- Output only a JSON object, nothing else

Output format:
{{
  "positive": "{seed_sentence}",
  "type1": "type 1 negative here",
  "type2": "type 2 negative here"
}}"""
def generate_contrastive_variants(
    seed_sentence: str,
    sector: str,
    max_retries: int = 3,
) -> dict | None:
    """
    Calls Groq API to generate Type 1 and Type 2 contrastive negatives.
    
    Args:
        seed_sentence: A real positive example from the labeled dataset
        sector: GICS sector of the source company
        max_retries: Number of retries on API failure or malformed output
    
    Returns:
        Dict with keys: positive, type1, type2.
        Returns None on failure.
    """
    prompt = build_contrastive_prompt(seed_sentence, sector)

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
            required_keys = {"positive", "type1", "type2"}
            if isinstance(result, dict) and required_keys.issubset(result.keys()):
                if all(isinstance(result[k], str) and len(result[k]) > 20
                       for k in ["type1", "type2"]):
                    return result

            print(f"Malformed output on attempt {attempt + 1}: {raw[:100]}")

        except json.JSONDecodeError:
            print(f"JSON parse error on attempt {attempt + 1}")
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")

        time.sleep(10)

    return None
def run_contrastive_generation(
    positives_path: str = "data/processed/train.parquet",
    output_path: str = "data/synthetic/contrastive_raw.parquet",
    max_seeds: int = None,
) -> None:
    """
    Full generation loop for contrastive hard negative augmentation.
    Iterates over all real positive examples, generates one Type 1 and
    one Type 2 contrastive negative per seed, and saves to parquet.
    
    Args:
        positives_path: Path to real positive examples
        output_path: Where to save raw contrastive outputs
        max_seeds: If set, limits generation for dry runs
    
    Notes:
        - Type 1: factual rewrite with negated claim
        - Type 2: factual rewrite asserting same claim without uncertainty
        - Checkpoint saved every 100 seeds to prevent data loss
        - Output includes provenance metadata for analysis
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

        result = generate_contrastive_variants(
            seed_sentence=seed,
            sector=sector,
        )

        if result is None:
            failed += 1
            continue

        # Type 1 record
        records.append({
            "sentence": result['type1'],
            "label": 0,
            "source": "synthetic_contrastive_type1",
            "seed_sentence": seed,
            "sector": sector,
            "contrastive_type": "type1",
        })

        # Type 2 record
        records.append({
            "sentence": result['type2'],
            "label": 0,
            "source": "synthetic_contrastive_type2",
            "seed_sentence": seed,
            "sector": sector,
            "contrastive_type": "type2",
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
    print(f"\nDone. Generated {len(df_out)} contrastive negatives. Failed: {failed}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    run_contrastive_generation(
        positives_path="data/processed/train.parquet",
        output_path="data/synthetic/contrastive_raw.parquet",
        max_seeds=5,
    )