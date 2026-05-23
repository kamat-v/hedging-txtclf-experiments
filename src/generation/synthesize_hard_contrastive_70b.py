# src/generation/synthesize_hard_contrastive_70b.py
# Hard contrastive negative generation using Llama 3.3 70B.
# Generates ONE high-quality hard negative per seed (vs two in 8B version)
# to stay within Groq free tier 500K TPD limit (674 seeds × ~400 tokens).
#
# Only Subtype A generated (epistemic verb reporting past fact) —
# the harder and more linguistically interesting subtype.
# Subtype B (modal describing external possibility) is more mechanical
# and less likely to benefit from 70B quality improvement.
#
# Output: data/synthetic/hard_contrastive_70b_raw.parquet
# Compare against: data/synthetic/hard_contrastive_raw.parquet (8B version)

import os
import json
import pandas as pd
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Upgraded to 70B for higher quality contrastive generation
MODEL = "llama-3.3-70b-versatile"

SECTORS = [
    "Technology", "Healthcare", "Financials", "Industrials",
    "Consumer Discretionary", "Energy", "Real Estate", "Materials"
]


def build_hard_contrastive_prompt_70b(seed_sentence: str, sector: str) -> str:
    """
    Single high-quality hard contrastive negative prompt for 70B model.
    Targets Subtype A only — epistemic verb reporting past fact.
    One variant per seed to stay within token budget.
    """
    return f"""You are generating training data for a hedging language classifier.
You are working with earnings call transcripts in the {sector} sector.

A hedged statement expresses the SPEAKER'S uncertainty about a future claim.
Example of genuine hedge: "We expect margins may decline next quarter."

A hard negative looks like a hedge on the surface but is NOT epistemically hedged.
Target type — Epistemic verb reporting past fact:
The verb (believe, think, expect) is present but describes something already known,
not uncertain. The speaker has no doubt about the claim.
Example: "We believe margins expanded last quarter." — reporting known fact, not uncertainty.

SEED: "{seed_sentence}"

Generate exactly ONE hard negative based on this seed:
- Use an epistemic verb (believe, think, expect, understand) to report a past fact
  or confirmed outcome related to the seed's topic
- Stay close to the seed's subject and vocabulary
- Do NOT express the speaker's own uncertainty about future outcomes
- Must sound like plausible earnings call language
- Output only a JSON object, nothing else

Output format:
{{
  "positive": "{seed_sentence}",
  "hard_negative": "epistemic verb reporting past fact here"
}}"""


def generate_hard_contrastive_70b(
    seed_sentence: str,
    sector: str,
    max_retries: int = 3,
) -> dict | None:
    """
    Calls Groq API with 70B model to generate one hard contrastive negative.

    Args:
        seed_sentence: A real positive example from the labeled dataset
        sector: GICS sector of the source company
        max_retries: Number of retries on API failure or malformed output

    Returns:
        Dict with keys: positive, hard_negative.
        Returns None on failure.
    """
    prompt = build_hard_contrastive_prompt_70b(seed_sentence, sector)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=300,  # single variant needs less headroom
            )

            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)

            required_keys = {"positive", "hard_negative"}
            if isinstance(result, dict) and required_keys.issubset(result.keys()):
                if isinstance(result["hard_negative"], str) and \
                   len(result["hard_negative"]) > 20:
                    return result

            print(f"Malformed output on attempt {attempt + 1}: {raw[:100]}")

        except json.JSONDecodeError:
            print(f"JSON parse error on attempt {attempt + 1}")
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")

        time.sleep(10)

    return None


def run_hard_contrastive_70b(
    positives_path: str = "data/processed/train.parquet",
    output_path: str = "data/synthetic/hard_contrastive_70b_raw.parquet",
    max_seeds: int = None,
) -> None:
    """
    Full generation loop — one 70B hard negative per real positive seed.
    Resume capability via checkpoint file.

    Args:
        positives_path: Path to training parquet
        output_path: Where to save raw outputs
        max_seeds: Cap for dry runs
    """
    os.makedirs("data/synthetic", exist_ok=True)
    checkpoint_path = output_path.replace('.parquet', '_checkpoint.parquet')

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        existing  = pd.read_parquet(checkpoint_path)
        completed = set(existing['seed_sentence'].unique())
        records   = existing.to_dict('records')
        print(f"Resuming: {len(existing)} variants already generated "
              f"({len(completed)} seeds completed).")
    else:
        completed = set()
        records   = []
        print("No checkpoint found — starting fresh.")

    positives = pd.read_parquet(positives_path)
    if 'label' in positives.columns:
        positives = positives[positives['label'] == 1].reset_index(drop=True)
    if max_seeds is not None:
        positives = positives.head(max_seeds)

    positives_remaining = positives[
        ~positives['sentence'].isin(completed)
    ].reset_index(drop=True)

    print(f"Total seeds: {len(positives)} | "
          f"Completed: {len(completed)} | "
          f"Remaining: {len(positives_remaining)}")

    failed = 0

    for i, row in positives_remaining.iterrows():
        seed   = row['sentence']
        sector = row['sector']

        result = generate_hard_contrastive_70b(
            seed_sentence=seed,
            sector=sector,
        )

        if result is None:
            failed += 1
            continue

        records.append({
            "sentence":      result['hard_negative'],
            "label":         0,
            "source":        "synthetic_hard_contrastive_70b",
            "seed_sentence": seed,
            "sector":        sector,
            "contrastive_type": "subtype_a_70b",
        })

        # Checkpoint every 100 seeds
        if (i + 1) % 100 == 0:
            df_checkpoint = pd.DataFrame(records)
            df_checkpoint.to_parquet(checkpoint_path, index=False)
            print(f"Processed {i + 1}/{len(positives_remaining)} remaining | "
                  f"Generated: {len(records)} total | Failed: {failed}")

        # Rate limit buffer — 70B uses more tokens per call
        time.sleep(10)

    df_out = pd.DataFrame(records)
    df_out.to_parquet(output_path, index=False)
    print(f"\nDone. Generated {len(df_out)} hard contrastive negatives. Failed: {failed}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    run_hard_contrastive_70b(
        positives_path="data/processed/train.parquet",
        output_path="data/synthetic/hard_contrastive_70b_raw.parquet",
        max_seeds=None,  
    )