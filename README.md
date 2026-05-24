# LLM Evaluation Toolkit

A lightweight toolkit for evaluating LLM and RAG pipelines using RAGAS, retrieval metrics, BLEU and ROUGE.

## Features
- Faithfulness and answer relevancy evaluation via RAGAS
- Retrieval metrics (precision@k, recall@k, MRR, hit rate)
- BLEU and ROUGE sentence-level scoring

## Requirements
- Python 3.8+
- Install required packages:

```bash
pip install datasets openai ragas langchain-openai langchain nltk rouge
```

Note: package names may vary by environment; adjust as needed.

## Setup
- The evaluation modules currently contain placeholders for API keys (`API_KEY`) and `BASE_URL` (see [Metrics/evaluation.py](Metrics/evaluation.py) and [Metrics/llm_evaluation.py](Metrics/llm_evaluation.py)). Replace these with your provider credentials or modify the code to read from environment variables.
- Optional: create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if you create one
```

## Usage

Example using the RAG + retrieval evaluation functions:

```python
from Metrics.llm_evaluation import (
    evaluate_pipeline,
)

question = "What is the capital of France?"
contexts = ["Paris is the capital of France."]
generated_answer = "Paris."
retrieved_ids = [101, 202, 303]
relevant_ids = [101]

result = evaluate_pipeline(
    question=question,
    contexts=contexts,
    generated_answer=generated_answer,
    retrieved_ids=retrieved_ids,
    relevant_ids=relevant_ids,
    retrieval_k=5,
)

print(result)
```

Examples for BLEU/ROUGE (from [Metrics/evaluation.py](Metrics/evaluation.py)):

```python
from Metrics.evaluation import evaluate_bleu, evaluate_rouge

bleu = evaluate_bleu(["the cat is on the mat"], "the cat is on the mat")
rouge = evaluate_rouge("the cat is on the mat", "the cat is on the mat")
```

## Files of interest
- [Metrics/evaluation.py](Metrics/evaluation.py): RAGAS, BLEU, and ROUGE helpers
- [Metrics/llm_evaluation.py](Metrics/llm_evaluation.py): faithfulness, hallucination, and retrieval metrics

## Contributing
- Improvements, bug fixes, and enhanced examples are welcome. Open a PR with changes.

## License
Apache 2.0
