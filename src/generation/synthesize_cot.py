# src/generation/synthesize_cot.py
# Chain-of-Thought variant of synthesize.py for positive augmentation.
# The only substantive change is in build_positive_prompt_cot — the model
# is asked to reason explicitly about hedging markers before generating variants.
# Generation loop, retry logic, and output schema are identical to synthesize.py.
#
# Updates from v1:
#   - max_tokens bumped to 600 to accommodate reasoning text before JSON array
#   - OUTPUT: delimiter added to prompt for robust JSON parsing
#   - Resume capability added — skips seeds already in checkpoint file

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


def build_positive_prompt_cot(seed_sentence: str, sector: str, persona: str) -> str:
    devices = random.sample(HEDGE_DEVICES, 2)

    return f"""You are {persona} in the {sector} sector speaking on an earnings call.

Below is a hedged statement from an earnings call transcript:

SEED: "{seed_sentence}"

Before generating variants, reason briefly about the hedging in the seed:
- Which hedging markers are present? (modals, cognitive verbs, conditionals, etc.)
- What epistemic function do they serve — uncertainty, conditionality, or caution?
- Which markers will you use in each variant, and why?

After reasoning, generate exactly 2 new hedged statements that preserve the epistemic
uncertainty of the original but vary in the following ways:
- Variant 1: shorter and more direct, use {devices[0]}, hedging marker near the beginning
- Variant 2: longer and more complex, use {devices[1]}, hedging marker embedded mid-sentence

Rules:
- Each variant must contain genuine epistemic uncertainty
- Do not copy the seed sentence — rewrite substantially
- Stay in the register of a {sector} sector earnings call
- Write your reasoning as plain text first
- Then output the JSON array on a new line starting with exactly "OUTPUT:"

Example output format:
The seed uses "we believe" (cognitive verb) expressing uncertainty about future outcomes.
Variant 1 will use an epistemic modal to front-load the hedge.
Variant 2 will embed a conditional clause mid-sentence to qualify the claim.
OUTPUT:
["variant 1 here", "variant 2 here"]"""


def generate_positive_variants_cot(
    seed_sentence: str,
    sector: str,
    persona: str = None,
    max_retries: int = 3,
) -> list[str]:
    """
    Calls Groq API to generate CoT-prompted positive hedge variants.
    Model reasons about hedging markers before generating variants.
    Reasoning is used only to condition generation — not returned or saved.

    Args:
        seed_sentence: A real positive example from the labeled dataset
        sector: GICS sector of the source company
        persona: Speaker role — if None, sampled randomly from PERSONAS
        max_retries: Number of retries on API failure or malformed output

    Returns:
        List of generated hedge variant strings, or empty list on failure.

    Notes:
        - Model outputs reasoning as plain text followed by OUTPUT: and JSON array
        - Parser splits on OUTPUT: and parses only what follows
        - Reasoning text is discarded after parsing
    """
    if persona is None:
        persona = random.choice(PERSONAS)

    prompt = build_positive_prompt_cot(seed_sentence, sector, persona)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=600,
            )
            raw = response.choices[0].message.content.strip()

            # Split on OUTPUT: delimiter and parse only what follows
            if "OUTPUT:" not in raw:
                print(f"No OUTPUT: delimiter found on attempt {attempt + 1}: {raw[:100]}")
                time.sleep(10)
                continue

            json_str = raw.split("OUTPUT:")[-1].strip()
            variants = json.loads(json_str)

            if isinstance(variants, list) and len(variants) == 2:
                if all(isinstance(v, str) and len(v) > 20 for v in variants):
                    return variants

            print(f"Malformed output on attempt {attempt + 1}: {raw[:100]}")

        except json.JSONDecodeError:
            print(f"JSON parse error on attempt {attempt + 1}")
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")

        time.sleep(10)

    return []


def run_positive_generation_cot(
    positives_path: str = "data/raw/positives.parquet",
    output_path: str = "data/synthetic/positive_raw_cot.parquet",
    variants_per_seed: int = 2,
    max_seeds: int = None,
) -> None:
    """
    Full generation loop for CoT positive augmentation with resume capability.
    Skips seeds already present in the checkpoint file so interrupted runs
    can be continued without regenerating from scratch.

    Args:
        positives_path: Path to real positive examples
        output_path: Where to save raw synthetic outputs
        variants_per_seed: Number of variants per seed (currently fixed at 2)
        max_seeds: Cap on seeds processed — set small for dry runs
    """
    os.makedirs("data/synthetic", exist_ok=True)

    checkpoint_path = output_path.replace('.parquet', '_checkpoint.parquet')

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        existing = pd.read_parquet(checkpoint_path)
        completed_seeds = set(existing['seed_sentence'].unique())
        records = existing.to_dict('records')
        print(f"Resuming from checkpoint: {len(existing)} sentences already generated "
              f"({len(completed_seeds)} seeds completed).")
    else:
        completed_seeds = set()
        records = []
        print("No checkpoint found — starting fresh.")

    positives = pd.read_parquet(positives_path)
    if 'label' in positives.columns:
        positives = positives[positives['label'] == 1].reset_index(drop=True)
    if max_seeds is not None:
        positives = positives.head(max_seeds)

    # Filter out already completed seeds
    positives_remaining = positives[
        ~positives['sentence'].isin(completed_seeds)
    ].reset_index(drop=True)

    print(f"Total seeds: {len(positives)} | "
          f"Already completed: {len(completed_seeds)} | "
          f"Remaining: {len(positives_remaining)}")

    failed = 0

    for i, row in positives_remaining.iterrows():
        seed = row['sentence']
        sector = row['sector']
        persona = random.choice(PERSONAS)

        variants = generate_positive_variants_cot(
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
                "source": "synthetic_positive_cot",
                "seed_sentence": seed,
                "sector": sector,
                "persona": persona,
            })

        if (i + 1) % 100 == 0:
            df_checkpoint = pd.DataFrame(records)
            df_checkpoint.to_parquet(checkpoint_path, index=False)
            print(f"Processed {i + 1}/{len(positives_remaining)} remaining seeds | "
                  f"Generated: {len(records)} total | Failed: {failed}")

    df_out = pd.DataFrame(records)
    df_out.to_parquet(output_path, index=False)
    print(f"\nDone. Generated {len(df_out)} synthetic positives total. Failed: {failed}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    run_positive_generation_cot(
        positives_path="data/processed/train.parquet",
        output_path="data/synthetic/positive_raw_cot.parquet",
    )