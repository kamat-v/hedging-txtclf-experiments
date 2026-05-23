# src/generation/synthesize_cot_70b.py
# 70B Chain-of-Thought positive augmentation.
# Generates ONE high-quality hedged variant per seed using Llama 3.3 70B.
# Model reasons explicitly about hedging markers before generating —
# CoT is still explicit since llama-3.3-70b-versatile is not reasoning-native.
#
# One variant per seed to stay within Groq 100K TPD limit.
# (~674 seeds × ~600 tokens = ~404K tokens, ~4 days at 100K/day)
#
# Output: data/synthetic/positive_cot_70b_raw.parquet

import os
import json
import pandas as pd
import time
import random
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

SECTORS = [
    "Technology", "Healthcare", "Financials", "Industrials",
    "Consumer Discretionary", "Energy", "Real Estate", "Materials"
]

HEDGE_DEVICES = [
    "epistemic modal verbs (may, might, could)",
    "cognitive verbs (we believe, we think, we expect)",
    "conditional framing (subject to, depending on, if)",
    "approximators (approximately, around, in the range of)",
    "explicit uncertainty markers (uncertain, unclear, difficult to predict)",
]


def build_cot_prompt_70b(seed_sentence: str, sector: str) -> str:
    """
    CoT positive augmentation prompt for 70B model.
    One high-quality hedged variant per seed.
    Reasoning precedes generation — OUTPUT: delimiter separates them.
    Persona and two-device constraint dropped — 70B doesn't need scaffolding.
    """
    device = random.choice(HEDGE_DEVICES)

    return f"""You are a financial analyst generating training data for a hedging language classifier.
You are working with earnings call transcripts in the {sector} sector.

A hedged statement expresses the SPEAKER'S genuine epistemic uncertainty about
a future outcome — not a past fact, not an external possibility, but the speaker's
own doubt about what will happen.

Below is a hedged statement from an earnings call transcript:

SEED: "{seed_sentence}"

Before generating a variant, reason briefly:
- Which hedging markers are present in the seed?
- What epistemic function do they serve?
- How will your variant express genuine uncertainty differently from the seed?
- You must use {device} as the primary hedging device.

After reasoning, generate exactly ONE new hedged statement that:
- Expresses genuine epistemic uncertainty about a future outcome
- Uses {device} as the primary hedging device
- Stays close to the seed's topic and vocabulary
- Does NOT copy the seed — rewrite substantially
- Sounds like plausible earnings call language

Write your reasoning as plain text first.
Then on a new line write exactly "OUTPUT:" followed by a JSON object.

Example format:
The seed uses "we expect" expressing forward-looking uncertainty.
My variant will use conditional framing.
OUTPUT:
{{"variant": "your hedged sentence here"}}"""


def generate_cot_variant_70b(
    seed_sentence: str,
    sector: str,
    max_retries: int = 3,
) -> list[str]:
    """
    Calls Groq 70B API to generate one CoT-prompted hedged variant.
    Reasoning conditions generation but is discarded after parsing.

    Args:
        seed_sentence: Real positive from labeled dataset
        sector: GICS sector of source company
        max_retries: Retries on API failure or malformed output

    Returns:
        List with one hedged variant string, or empty list on failure.
    """
    prompt = build_cot_prompt_70b(seed_sentence, sector)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=500,
            )
            raw = response.choices[0].message.content.strip()

            # Split on OUTPUT: delimiter — reasoning precedes, JSON follows
            if "OUTPUT:" not in raw:
                print(f"No OUTPUT: delimiter on attempt {attempt + 1}: {raw[:100]}")
                time.sleep(10)
                continue

            json_str = raw.split("OUTPUT:")[-1].strip()
            result = json.loads(json_str)

            if isinstance(result, dict) and "variant" in result:
                if isinstance(result["variant"], str) and len(result["variant"]) > 20:
                    return [result["variant"]]

            print(f"Malformed output on attempt {attempt + 1}: {raw[:100]}")

        except json.JSONDecodeError:
            print(f"JSON parse error on attempt {attempt + 1}")
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")

        time.sleep(10)

    return []


def run_cot_generation_70b(
    positives_path: str = "data/processed/train.parquet",
    output_path: str = "data/synthetic/positive_cot_70b_raw.parquet",
    max_seeds: int = None,
) -> None:
    """
    Full generation loop with resume capability.
    One 70B CoT variant per real positive seed.

    Args:
        positives_path: Path to training parquet
        output_path: Where to save raw outputs
        max_seeds: Cap for dry runs
    """
    os.makedirs("data/synthetic", exist_ok=True)
    checkpoint_path = output_path.replace('.parquet', '_checkpoint.parquet')

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        existing       = pd.read_parquet(checkpoint_path)
        completed_seeds = set(existing['seed_sentence'].unique())
        records        = existing.to_dict('records')
        print(f"Resuming: {len(existing)} variants already generated "
              f"({len(completed_seeds)} seeds completed).")
    else:
        completed_seeds = set()
        records         = []
        print("No checkpoint found — starting fresh.")

    positives = pd.read_parquet(positives_path)
    if 'label' in positives.columns:
        positives = positives[positives['label'] == 1].reset_index(drop=True)
    if max_seeds is not None:
        positives = positives.head(max_seeds)

    positives_remaining = positives[
        ~positives['sentence'].isin(completed_seeds)
    ].reset_index(drop=True)

    print(f"Total seeds: {len(positives)} | "
          f"Completed: {len(completed_seeds)} | "
          f"Remaining: {len(positives_remaining)}")

    failed = 0

    for i, row in positives_remaining.iterrows():
        seed   = row['sentence']
        sector = row['sector']

        variants = generate_cot_variant_70b(
            seed_sentence=seed,
            sector=sector,
        )

        if not variants:
            failed += 1
            continue

        records.append({
            "sentence":      variants[0],
            "label":         1,
            "source":        "synthetic_positive_cot_70b",
            "seed_sentence": seed,
            "sector":        sector,
        })

        if (i + 1) % 100 == 0:
            df_checkpoint = pd.DataFrame(records)
            df_checkpoint.to_parquet(checkpoint_path, index=False)
            print(f"Processed {i + 1}/{len(positives_remaining)} remaining | "
                  f"Generated: {len(records)} total | Failed: {failed}")

        # Rate limit buffer — 70B uses more tokens per call
        time.sleep(3)

    df_out = pd.DataFrame(records)
    df_out.to_parquet(output_path, index=False)
    print(f"\nDone. Generated {len(df_out)} CoT positive variants. Failed: {failed}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    # Set max_seeds=5 for dry run first
    run_cot_generation_70b(
        positives_path="data/processed/train.parquet",
        output_path="data/synthetic/positive_cot_70b_raw.parquet",
        max_seeds=None,
    )