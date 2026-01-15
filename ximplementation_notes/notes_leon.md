# Notes of relevant aspects of implementation

## LLM part
- DEFAULT_MAX_CONTEXT_CHARS in llm_config was at 2000 which is a major mismatch with max chunk size of 2500 and default of 5 chunks of context -> changed to 100000

## Evaluation
- Parameters for eval:
    - use bert model (qwen is too slow)
    - document top_k used for eval
    - also document MAX_CHUNK_SIZE = 2500, OVERLAP_SIZE = 200 (and KEEP at these values)
- Use actual updated chunks
- 1 or two examples with not just 1 paper; synthesis of 2 papers; (different difficulties)
    - Evaluate with facts from just one paper; is the paper found?
        - Make x (e.g. 5) example queries
        - Determine "ground truth"; what paper/chunk should be found?
        - Calculate metric (ground truth against model)

## App constraints
- authors are not passed to the LLM, so cannot reference authors in the user query
- paper titles are passed to the prompt, BUT are often placeholders like "article" since they are not optimally handled in the extraction logic
    - --> paper titles can not be reliably referenced in the user query
- Improved chunking and pdf extraction constraints
    - tables and figures are not extracted well
    - pdf unicode signs (/uniFB01) are not converted correctly and make it into the LLM prompt

