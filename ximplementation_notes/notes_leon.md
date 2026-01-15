# Notes of relevant aspects of implementation

## Chunking
- improved chunking in a separate notebook (left backend as-is for now)
- [TODO]: test more: look at all the chunks of all papers; is it sensible?

## [TODO] RAG pipeline via notebook (rag_pipeline.ipynb)
- is chunking logic consistent with the one tested in chunking_evaluation.ipynb?
- **Improve until it can be used well for prototyping and testing**
    - debug errors and wrong behaviour
    - improve usability
        - make it easier to see the actual and whole retrieved chunks
        - IS CHUNK SIZE TOO SMALL?
    - get a feeling for how well it works
    - try scientific embedding model
        - chunk size and embedding model compatible?
    - add modes

## LLM part
- prompt template seems nonsensical
- understand how this works; improve

## Evaluation
- Parameters for eval:
    - use bert model (qwen is too slow)
    - document top_k used for eval
- Use actual updated chunks
- Evaluate with facts from just one paper; is the paper found?
    - Make x (e.g. 5) example queries
    - Determine "ground truth"; what paper/chunk should be found?
    - Calculate metric (ground truth against model)
- 1 or two examples with not just 1 paper; synthesis of 2 papers; (different difficulties)
