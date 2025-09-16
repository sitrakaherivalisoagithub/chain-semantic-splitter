# -*- coding: utf-8 -*-
"""
This module defines the SemanticCharacterTextSplitter class for semantically
aware text splitting within the LangChain ecosystem.
"""

import logging
import json
import time
from typing import List, Any, Dict, Union

from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
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
    boundaries in the text.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_retries: int = 3,
        separator: str = "\n",
        return_docs: bool = False,
        **kwargs: Any,
    ):
        """
        Initializes the SemanticCharacterTextSplitter.

        Args:
            llm (BaseLanguageModel): An instance of a LangChain-compatible LLM
                (e.g., ChatGoogleGenerativeAI) that will be used to make merge
                decisions.
            chunk_size (int): The target size for the initial character-based
                splitting.
            chunk_overlap (int): The overlap between chunks in the initial
                splitting.
            max_retries (int): The maximum number of times to retry an API call
                to the LLM in case of failure.
            separator (str): The separator to use when merging chunks.
            return_docs (bool): If True, returns a list of LangChain Document
                objects; otherwise, returns a list of strings.
            **kwargs: Additional keyword arguments to be passed to the parent
                TextSplitter class.
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.llm = llm
        self.max_retries = max_retries
        self._separator = separator
        self.return_docs = return_docs

        self._initial_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
        )
        self._decision_chain = self._build_decision_chain()

    def _build_decision_chain(self):
        """
        Constructs the LangChain Expression Language (LCEL) chain for making
        merge decisions.

        The chain consists of a prompt, the language model, and a JSON parser.
        """
        parser = JsonOutputParser(pydantic_object=MergeDecision)

        prompt = ChatPromptTemplate.from_template(
            """You are an expert in text analysis and segmentation. Your task is to determine if two adjacent text chunks should be merged into a single, coherent segment.

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
Consider if the second chunk continues the thought, topic, or narrative of the first.

{format_instructions}
"""
        )

        return (prompt | self.llm | parser).with_partial(
            format_instructions=parser.get_format_instructions()
        )

    def _decide_to_merge(self, chunk1: str, chunk2: str) -> bool:
        """
        Uses the LLM chain to decide whether two chunks should be merged.

        This method includes retry logic, exponential backoff, and a fallback mechanism.

        Args:
            chunk1 (str): The current text buffer.
            chunk2 (str): The next chunk to potentially merge.

        Returns:
            bool: True if the chunks should be merged, False otherwise.
        """
        chunk1_end = chunk1[-(self._chunk_overlap + 100):]
        chunk2_start = chunk2[:(self._chunk_overlap + 100)]

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
            time.sleep(2 ** attempt)

        logger.error(f"Failed to get a valid decision from LLM after {self.max_retries} attempts. Applying fallback strategy: not merging.")
        return False

    def split_text(self, text: str) -> Union[List[str], List[Document]]:
        """
        Splits a given text into semantically coherent chunks.

        Args:
            text (str): The text to be split.

        Returns:
            Union[List[str], List[Document]]: A list of strings or LangChain
            Document objects, where each item is a semantically coherent chunk.
        """
        initial_chunks = self._initial_splitter.split_text(text)
        if not initial_chunks:
            return []

        final_chunks = []
        current_chunk_buffer = initial_chunks[0]

        for i in range(1, len(initial_chunks)):
            next_chunk = initial_chunks[i]
            if self._decide_to_merge(current_chunk_buffer, next_chunk):
                current_chunk_buffer += self._separator + next_chunk
            else:
                final_chunks.append(current_chunk_buffer)
                current_chunk_buffer = next_chunk

        if current_chunk_buffer:
            final_chunks.append(current_chunk_buffer)

        if self.return_docs:
            return [Document(page_content=chunk) for chunk in final_chunks]
        return final_chunks
