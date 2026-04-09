"""LLM assistant module for RAG-based QA."""

import time
from typing import Dict, List, Optional

from huggingface_hub import InferenceClient

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert, truthful assistant for Devsinc company policies. "
    "Answer ONLY using the provided policy context. "
    "If the information is not in the context, say: "
    "'I could not find this information in the official policies.' "
    "Always be professional, concise, and cite the exact section/title when possible."
)


class RAGAssistant:
    """
    RAG-powered assistant that uses Weaviate retrieval plus LLM generation.
    """

    def __init__(
        self,
        rag_pipeline,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct",
        hf_token: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.90,
        default_num_results: int = 4,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.5,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        """
        Initialize assistant with retrieval pipeline and HF inference client.

        Args:
            rag_pipeline: Retrieval pipeline used to fetch relevant chunks.
            model_name: Hugging Face model id for answer generation.
            hf_token: HF API token for authenticated inference requests.
            temperature: Sampling temperature for generation variability.
            max_tokens: Maximum tokens to generate in a response.
            top_p: Nucleus sampling value for token selection.
            default_num_results: Default number of retrieved chunks in context.
            max_retries: Max retry attempts for failed LLM API requests.
            retry_backoff_seconds: Base backoff seconds for retry delays.
            system_prompt: Default system instruction prompt for the assistant.
        """
        self.rag_pipeline = rag_pipeline
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.default_num_results = default_num_results
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.system_prompt = system_prompt
        self.client = InferenceClient(model=self.model_name, token=hf_token)
        print(f"[STEP] RAGAssistant initialized with model: {model_name}")

    def _build_rag_context(self, query: str, num_results: int = 4) -> str:
        """
        Retrieve top reranked chunks and compose context block.
        """
        results: List[Dict] = self.rag_pipeline.query(query, limit=num_results)
        context_parts = []
        for index, res in enumerate(results, start=1):
            meta = res["metadata"]
            source = f"{meta['title']} -> {meta['heading']}"
            context_parts.append(f"[Source {index}] {source}\n{meta['text']}")
        return "\n\n" + "=" * 60 + "\n\n".join(context_parts)

    def answer(
        self,
        user_query: str,
        num_results: Optional[int] = None,
        custom_system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Answer question using retrieved policy context and chat completion.
        """
        effective_num_results = num_results or self.default_num_results
        context = self._build_rag_context(user_query, num_results=effective_num_results)
        system_prompt = custom_system_prompt or self.system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Here is the relevant context from official Devsinc policy documents:\n\n"
                    f"{context}\n\n"
                    f"Question: {user_query}\n\n"
                    "Please provide a clear, accurate answer based only on the context above."
                ),
            },
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                    top_p=self.top_p,
                    stream=False,
                )
                return response.choices[0].message.content.strip()
            except Exception as error:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"HF chat_completion failed after {self.max_retries} attempts: {error}"
                    ) from error
                wait_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"[WARN] HF chat attempt {attempt} failed: {error}. "
                    f"Retrying in {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)

        return ""

