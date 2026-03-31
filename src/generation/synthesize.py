# src/generation/synthesize.py
# LLM-based synthetic data generation for hedging language classification.
# Supports two augmentation conditions:
#   1. Positive augmentation — generates hedge variants of real positive examples
#   2. Contrastive augmentation — generates hard negatives from real positive examples
#
# Generation uses the Groq API (Llama 3.1 8B Instruct) with structured prompts
# incorporating persona conditioning, verbalized sampling, sector grounding,
# and sentence length variation to maximize diversity.
#
# Both global and error-driven augmentation are supported — the only difference
# is which seed sentences are passed in at call time.

import os
import json
import pandas as pd
import time
import random
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Generation model — start with 8B for speed during prompt development
# Switch to llama-3.3-70b-versatile for final generation pass if needed
MODEL = "llama-3.1-8b-instant"
# --- Persona and sector context ---
# CEO tends to hedge on strategy and vision
# CFO tends to hedge on financials and guidance
# IR Officer tends to hedge on market conditions and outlook

PERSONAS = [
    "a CEO discussing strategic outlook",
    "a CFO discussing financial guidance",
    "an Investor Relations officer discussing market conditions",
]

HEDGE_DEVICES = [
    "epistemic modal verbs (may, might, could)",
    "cognitive verbs (we believe, we think, we expect)",
    "conditional framing (subject to, depending on, if)",
    "approximators (approximately, around, in the range of)",
    "explicit uncertainty markers (uncertain, unclear, difficult to predict)",
]

SECTORS = [
    "Technology", "Healthcare", "Financials", "Industrials",
    "Consumer Discretionary", "Energy", "Real Estate", "Materials"
]


def build_positive_prompt(seed_sentence: str, sector: str, persona: str) -> str:
    # Sample two distinct hedge devices for the two variants
    devices = random.sample(HEDGE_DEVICES, 2)
    
    return f"""You are {persona} in the {sector} sector speaking on an earnings call.

Below is a hedged statement from an earnings call transcript:

SEED: "{seed_sentence}"

Generate exactly 2 new hedged statements that preserve the epistemic uncertainty
of the original but vary in the following ways:
- Variant 1: shorter and more direct, use {devices[0]}, hedging marker near the beginning
- Variant 2: longer and more complex, use {devices[1]}, hedging marker embedded mid-sentence

Rules:
- Each variant must contain genuine epistemic uncertainty
- Do not copy the seed sentence — rewrite substantially
- Stay in the register of a {sector} sector earnings call
- Output only a JSON array of 2 strings, nothing else

Example output format:
["variant 1 here", "variant 2 here"]"""

def generate_positive_variants(
    seed_sentence: str,
    sector: str,
    persona: str = None,
    max_retries: int = 3,
) -> list[str]:
    """
    Calls Groq API to generate positive hedge variants for a seed sentence.
    
    Args:
        seed_sentence: A real positive example from the labeled dataset
        sector: GICS sector of the source company — used for grounding
        persona: Speaker role — if None, sampled randomly from PERSONAS
        max_retries: Number of retries on API failure or malformed output
    
    Returns:
        List of generated hedge variant strings, or empty list on failure.
    
    Notes:
        - Output is expected as a JSON array of 2 strings
        - Malformed JSON is caught and retried up to max_retries times
        - A small delay between retries avoids rate limit issues
    """
    if persona is None:
        persona = random.choice(PERSONAS)

    prompt = build_positive_prompt(seed_sentence, sector, persona)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON output — model instructed to return only a JSON array
            variants = json.loads(raw)

            # Validate output structure
            if isinstance(variants, list) and len(variants) == 2:
                if all(isinstance(v, str) and len(v) > 20 for v in variants):
                    return variants

            print(f"Malformed output on attempt {attempt + 1}: {raw[:100]}")

        except json.JSONDecodeError:
            print(f"JSON parse error on attempt {attempt + 1}")
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")

        time.sleep(10)  # brief pause before retry

    return []  # return empty list if all retries fail

def run_positive_generation(
    positives_path: str = "data/raw/positives.parquet",
    output_path: str = "data/synthetic/positive_raw.parquet",
    variants_per_seed: int = 2,
    max_seeds: int= None, #set to a small number for dry runs
) -> None:
    """
    Full generation loop for positive augmentation.
    Iterates over all real positive examples, generates variants for each,
    and saves raw (unfiltered) synthetic positives to parquet.
    
    Args:
        positives_path: Path to real positive examples
        output_path: Where to save raw synthetic outputs
        variants_per_seed: Number of variants to generate per seed sentence
    
    Notes:
        - Sector metadata from the real positives is used for grounding
        - Persona is sampled randomly per seed for diversity
        - Failed generations are skipped and logged
        - Output includes provenance metadata for analysis
    """
    os.makedirs("data/synthetic", exist_ok=True)

    positives = pd.read_parquet(positives_path)
    if 'label' in positives.columns:
        positives=positives[positives['label']==1].reset_index(drop=True)
    if max_seeds is not None:
       positives = positives.head(max_seeds)
    print(f"Loaded {len(positives)} seed positives.")

    records = []
    failed = 0

    for i, row in positives.iterrows():
        seed = row['sentence']
        sector = row['sector']
        persona = random.choice(PERSONAS)

        variants = generate_positive_variants(
            seed_sentence=seed,
            sector=sector,
            persona=persona,
        )

        if not variants:
            failed += 1
            continue

        for v in variants:
            records.append({
                "sentence": v,
                "label": 1,
                "source": "synthetic_positive",
                "seed_sentence": seed,
                "sector": sector,
                "persona": persona,
            })
        # Save checkpoint and print progress every 100 seeds
        if (i + 1) % 100 == 0:
            df_checkpoint = pd.DataFrame(records)
            df_checkpoint.to_parquet(output_path.replace('.parquet', '_checkpoint.parquet'), index=False)
            print(f"Processed {i + 1}/{len(positives)} seeds | "
                  f"Generated: {len(records)} | Failed: {failed}")

    df_out = pd.DataFrame(records)
    df_out.to_parquet(output_path, index=False)
    print(f"\nDone. Generated {len(df_out)} synthetic positives. Failed: {failed}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    run_positive_generation(
        positives_path="data/processed/train.parquet",
        output_path="data/synthetic/positive_raw.parquet",
    )