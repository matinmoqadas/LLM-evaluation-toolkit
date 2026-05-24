from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness
from langchain_openai import ChatOpenAI


API_KEY = "aa"
BASE_URL = "https://api.avalai.ir/v1"


def get_eval_llm() -> LangchainLLMWrapper:
    """Initializes the evaluation language model."""
    return LangchainLLMWrapper(
        ChatOpenAI(base_url=BASE_URL, model="gpt-5-mini", api_key=API_KEY)
    )


def create_embedding(text: str) -> list[float]:
    """Creates an embedding vector for a single string."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    return resp.data[0].embedding


class CustomEmbeddingModel:
    """Minimal embedding wrapper compatible with RAGAS."""

    def embed_query(self, text: str) -> list[float]:
        return create_embedding(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [create_embedding(t) for t in texts]


def get_eval_embeddings() -> LangchainEmbeddingsWrapper:
    """Initializes the evaluation embedding model."""
    return LangchainEmbeddingsWrapper(CustomEmbeddingModel())


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    mrr_at_k: float
    num_relevant: int
    num_retrieved: int
    num_relevant_found: int


def compute_retrieval_metrics(
    retrieved_items: Sequence[Union[str, int]],
    relevant_items: Sequence[Union[str, int]],
    k: int = 10,
) -> RetrievalMetrics:
    """Computes standard retrieval evaluation metrics for top-k retrieval."""
    retrieved_topk = list(retrieved_items)[:k]
    relevant_set = set(relevant_items)
    num_relevant = len(relevant_set)
    num_retrieved = len(retrieved_topk)

    relevant_found = [item for item in retrieved_topk if item in relevant_set]
    num_relevant_found = len(relevant_found)

    precision_at_k = num_relevant_found / num_retrieved if num_retrieved else 0.0
    recall_at_k = num_relevant_found / num_relevant if num_relevant else 0.0

    mrr = 0.0
    for rank, item in enumerate(retrieved_topk, start=1):
        if item in relevant_set:
            mrr = 1.0 / rank
            break

    hit_rate_at_k = 1.0 if num_relevant_found > 0 else 0.0

    return RetrievalMetrics(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        hit_rate_at_k=hit_rate_at_k,
        mrr_at_k=mrr,
        num_relevant=num_relevant,
        num_retrieved=num_retrieved,
        num_relevant_found=num_relevant_found,
    )


def evaluate_faithfulness(
    question: str,
    contexts: Sequence[str],
    generated_answer: str,
    reference_answer: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluates faithfulness of an LLM answer against retrieved context."""
    data = {
        "question": [question],
        "contexts": [list(contexts)],
        "answer": [generated_answer],
    }
    if reference_answer is not None:
        data["ground_truth"] = [reference_answer]

    ds = Dataset.from_dict(data)
    llm = get_eval_llm()
    emb = get_eval_embeddings()
    metrics = [Faithfulness(llm=llm)]

    result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb, show_progress=False)
    output = result.to_pandas().iloc[0].to_dict()
    return {
        "faithfulness": output.get("faithfulness", 0.0),
        "answer_relevancy": output.get("answer_relevancy", 0.0),
    }


def evaluate_hallucination(
    question: str,
    contexts: Sequence[str],
    generated_answer: str,
    reference_answer: Optional[str] = None,
    faithfulness_threshold: float = 0.75,
) -> Dict[str, Union[float, bool]]:
    """Detects hallucination risk by measuring faithfulness and applying a threshold."""
    scores = evaluate_faithfulness(
        question=question,
        contexts=contexts,
        generated_answer=generated_answer,
        reference_answer=reference_answer,
    )
    faithfulness_score = scores.get("faithfulness", 0.0)
    hallucination_score = 1.0 - faithfulness_score
    return {
        "faithfulness": faithfulness_score,
        "hallucination_score": hallucination_score,
        "is_hallucinating": faithfulness_score < faithfulness_threshold,
        "faithfulness_threshold": faithfulness_threshold,
    }


def evaluate_pipeline(
    question: str,
    contexts: Sequence[str],
    generated_answer: str,
    retrieved_ids: Optional[Sequence[Union[str, int]]] = None,
    relevant_ids: Optional[Sequence[Union[str, int]]] = None,
    reference_answer: Optional[str] = None,
    retrieval_k: int = 10,
) -> Dict[str, object]:
    """Runs a combined evaluation for faithfulness, hallucination, and retrieval."""
    evaluation: Dict[str, object] = {
        "faithfulness": evaluate_faithfulness(
            question=question,
            contexts=contexts,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
        ),
        "hallucination": evaluate_hallucination(
            question=question,
            contexts=contexts,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
        ),
    }

    if retrieved_ids is not None and relevant_ids is not None:
        evaluation["retrieval"] = compute_retrieval_metrics(
            retrieved_items=retrieved_ids,
            relevant_items=relevant_ids,
            k=retrieval_k,
        ).__dict__

    return evaluation
