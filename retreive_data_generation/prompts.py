GENERATE_PROMPT = """<|begin_of_text|>
### Instruction
You're an AI tasked with turning medical symptoms or conditions into a single, natural patient-style question.
You need to generate query so that both patient_score and relevance_score are close to 1.0.

### Context
{context}

### Generate Query output:
Return only ONE patient-style question in natural language. No explanations, no headings, no extra formatting.
<|end_of_text|>
"""

REGENERATE_PROMPT = """<|begin_of_text|>
### Instruction
You're an AI tasked with turning medical symptoms or conditions into a single, natural patient-style question.
You need to generate query so that both patient_score and relevance_score are close to 1.0.

### Context
{context}

### Generated Query
{query}

Note: The generated query did not meet the following conditions (patient_score: {patient_score}, relevance_score: {relevance_score}).
Please regenerate it so that satisfies the conditions.

### Generate Query output:
Return only ONE patient-style question in natural language. No explanations, no headings, no extra formatting.
<|end_of_text|>
"""

EVALUATE_PROMPT = """<|begin_of_text|>
### Instruction
You are an evaluator. For the given query and context, provide two numeric scores between 0 and 1:

1. **Patient-style score**: Does this question sound like something a patient would naturally ask?
2. **Relevance score**: How relevant is this question to the original medical context?

Return format (JSON only):
{{
  "patient_score": float between 0 and 1,
  "relevance_score": float between 0 and 1
}}


### Context
{context}

### Generated Query
{query}

Now return the scores in JSON format only:
<|end_of_text|>
"""


def generate_prompt(
    prompt_type: str,
):
    if prompt_type == 'generate':
        return GENERATE_PROMPT
    elif prompt_type == 'regenerate':
        return REGENERATE_PROMPT
    elif prompt_type == 'evaluate':
        return EVALUATE_PROMPT
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
