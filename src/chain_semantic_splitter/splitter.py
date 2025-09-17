# -*- coding: utf-8 -*-
"""
This module defines the SemanticCharacterTextSplitter class for semantically
aware text splitting within the LangChain ecosystem.
"""
from __future__ import annotations

import logging
import json
import time
from typing import Any, Dict

from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.documents import Document

# Configure a logger for warnings and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MergeDecision(BaseModel):
    """
    A Pydantic model representing the structured output for the merge decision.
    This ensures the LLM provides a predictable JSON format.
    """
    should_merge: bool = Field(description="True if the two text chunks should be merged, False otherwise.")
    reason: str = Field(description="A brief justification for the merge decision.")


class SemanticCharacterTextSplitter(TextSplitter):
    """
    A text splitter that uses a language model to make semantic decisions
    about how to split a document.

    This splitter first performs a rough pre-splitting using a standard
    character-based method, and then iteratively decides whether adjacent
    chunks should be merged based on their semantic content.

    This approach aims to create more coherent chunks that respect logical
    boundaries in the text, and offers flexibility in its configuration.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        initial_splitter: TextSplitter | None = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 400,
        context_margin: int = 100,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """
        Initializes the SemanticCharacterTextSplitter.

        Args:
            llm (BaseLanguageModel): An instance of a LangChain-compatible LLM
                (e.g., ChatGoogleGenerativeAI) used to make merge decisions.
            initial_splitter (TextSplitter, optional): A pre-configured text
                splitter for the initial coarse splitting. If None, a
                RecursiveCharacterTextSplitter will be created using
                chunk_size and chunk_overlap. Defaults to None.
            chunk_size (int): The target size for the initial splitting if no
                initial_splitter is provided.
            chunk_overlap (int): The overlap for the initial splitting if no
                initial_splitter is provided.
            context_margin (int): The number of extra characters to include
                on each side of the overlap zone when presenting context
                to the LLM.
            max_retries (int): The maximum number of times to retry an API call
                to the LLM in case of failure.
            **kwargs: Additional keyword arguments for the parent TextSplitter class.
        """
        # Note: we pass chunk_size and chunk_overlap to super() for LangChain compatibility,
        # but the primary logic uses the properties of _initial_splitter.
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

        self.llm = llm
        self.context_margin = context_margin
        self.max_retries = max_retries
        
        if initial_splitter:
            self._initial_splitter = initial_splitter
            # Override chunk_size and chunk_overlap to match the provided splitter
            self._chunk_size = initial_splitter._chunk_size
            self._chunk_overlap = initial_splitter._chunk_overlap
        else:
            self._initial_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap
            )

    @property
    def _decision_chain(self):
        """
        Lazily builds and returns the LCEL chain for merge decisions.
        This ensures the chain always uses the current state of the object (e.g., self.llm).
        """
        if not hasattr(self, "_internal_decision_chain"):
            self._internal_decision_chain = self._build_decision_chain()
        return self._internal_decision_chain

    def _build_decision_chain(self):
        """
        Constructs the LangChain Expression Language (LCEL) chain for making
        merge decisions.
        """
        parser = JsonOutputParser(pydantic_object=MergeDecision)

        prompt = ChatPromptTemplate.from_template(
            """You are an expert in text analysis and segmentation for a Retrieval-Augmented Generation (RAG) system. Your task is to be strict and determine if two adjacent text chunks should be merged.

Only merge if the second chunk is a direct and necessary continuation of the first. If a new, distinct topic or sub-topic begins, do not merge.

Context:
The first chunk ends with the following text:
---
{chunk1_end}
---

The second chunk begins with the following text:
---
{chunk2_start}
---

Decision Task:
Based on the content, do these two chunks belong together in the same semantic segment?

{format_instructions}
"""
        ).partial(format_instructions=parser.get_format_instructions())

        return prompt | self.llm | parser

    def _decide_to_merge(self, chunk1: str, chunk2: str) -> bool:
        """
        Uses the LLM chain to decide whether two chunks should be merged.
        This method includes retry logic, exponential backoff, and a fallback mechanism.
        """
        context_window_size = self._chunk_overlap + self.context_margin
        chunk1_end = chunk1[-context_window_size:]
        chunk2_start = chunk2[:context_window_size]

        for attempt in range(self.max_retries):
            try:
                decision: Dict[str, Any] = self._decision_chain.invoke({
                    "chunk1_end": chunk1_end,
                    "chunk2_start": chunk2_start,
                })
                should_merge = decision.get("should_merge", False)
                reason = decision.get("reason", "No reason provided")
                logger.info(f"LLM decision (attempt {attempt+1}): should_merge={should_merge}, reason='{reason}'")
                return should_merge
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt+1}/{self.max_retries}: JSON parse error: {e}. Retrying...")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{self.max_retries}: Unexpected API error: {e}. Retrying...")

            # Exponential backoff before the next retry
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        logger.error(f"Failed to get a valid decision from LLM after {self.max_retries} attempts. Applying fallback strategy: not merging.")
        return False

    def split_text(self, text: str) -> list[str]:
        """
        Splits a given text into semantically coherent chunks.
        """
        initial_chunks = self._initial_splitter.split_text(text)
        if not initial_chunks:
            return []

        final_chunks = []
        current_chunk_buffer = initial_chunks[0]

        for i in range(1, len(initial_chunks)):
            next_chunk = initial_chunks[i]
            if self._decide_to_merge(current_chunk_buffer, next_chunk):
                # Only append the part of the next_chunk that is not overlapping
                current_chunk_buffer += next_chunk[self._chunk_overlap:]
            else:
                final_chunks.append(current_chunk_buffer)
                current_chunk_buffer = next_chunk

        if current_chunk_buffer:
            final_chunks.append(current_chunk_buffer)

        return final_chunks

    def create_documents(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = _metadatas[i].copy()
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

