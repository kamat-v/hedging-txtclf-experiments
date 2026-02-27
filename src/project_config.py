# Task Definition
TASK = "Sentence-level binary classification of hedging language in earnings call transcripts"

# Label Schema
LABEL_POSITIVE = 1  # Hedged forward-looking statement
LABEL_NEGATIVE = 0  # Confident, factual, or non-hedging statement

# Hedging Definition

HEDGING_DEFINITION = """
A hedged statement expresses uncertainty, conditionality, or epistemic caution about a claim 
(often, but not only, future outcomes). Indicators include epistemic modals (may,might,could), 
conditional language (if, unless, subject to), and uncertainty markers 
(we believe, expect, think, likely), especially when the speaker is qualifying confidence rather 
than stating a settled fact. 
"""