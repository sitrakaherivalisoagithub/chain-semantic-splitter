# -*- coding: utf-8 -*-
"""
This module defines the SemanticCharacterTextSplitter class for semantically
aware text splitting within the LangChain ecosystem.
"""

import logging
import json
from typing import List, Any, Dict

from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models.base import BaseLanguageModel

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
            **kwargs: Additional keyword arguments to be passed to the parent
                TextSplitter class.
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.llm = llm
        self.max_retries = max_retries
        self._initial_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._decision_chain = self._build_decision_chain()

    def _build_decision_chain(self):
        """
        Constructs the LangChain Expression Language (LCEL) chain for making
        merge decisions.

        The chain consists of a prompt, the language model, and a JSON parser.
        """
        # 1. Create a parser for the structured JSON output
        parser = JsonOutputParser(pydantic_object=MergeDecision)

        # 2. Define the prompt template
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

        # 3. Assemble the chain
        chain = (
            prompt
            | self.llm
            | parser
        )
        # Inject the format instructions into the chain's prompt
        return chain.with_partial(format_instructions=parser.get_format_instructions())

    def _decide_to_merge(self, chunk1: str, chunk2: str) -> bool:
        """
        Uses the LLM chain to decide whether two chunks should be merged.

        This method includes retry logic and a fallback mechanism.

        Args:
            chunk1 (str): The current text buffer.
            chunk2 (str): The next chunk to potentially merge.

        Returns:
            bool: True if the chunks should be merged, False otherwise.
        """
        # Use the end of the first chunk and the start of the second for efficiency
        chunk1_end = chunk1[-(self._chunk_overlap + 100):]
        chunk2_start = chunk2[:(self._chunk_overlap + 100)]

        for attempt in range(self.max_retries):
            try:
                decision: Dict[str, Any] = self._decision_chain.invoke({
                    "chunk1_end": chunk1_end,
                    "chunk2_start": chunk2_start
                })
                return decision.get('should_merge', False)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries}: Failed to parse JSON response. Error: {e}. Retrying..."
                )
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries}: An unexpected API error occurred: {e}. Retrying..."
                )
        
        # Fallback strategy after all retries have failed
        logger.error(
            f"Failed to get a valid decision from LLM after {self.max_retries} attempts. "
            "Applying fallback strategy: not merging."
        )
        return False

    def split_text(self, text: str) -> List[str]:
        """
        Splits a given text into semantically coherent chunks.

        Args:
            text (str): The text to be split.

        Returns:
            List[str]: A list of strings, where each string is a
                       semantically coherent chunk.
        """
        # 1. Perform initial coarse splitting
        initial_chunks = self._initial_splitter.split_text(text)

        if not initial_chunks:
            return []

        # 2. Iteratively merge chunks based on semantic coherence
        final_chunks = []
        current_chunk_buffer = initial_chunks[0]

        for i in range(1, len(initial_chunks)):
            next_chunk = initial_chunks[i]

            if self._decide_to_merge(current_chunk_buffer, next_chunk):
                # If the decision is to merge, append the next chunk to the buffer
                current_chunk_buffer += self._separator + next_chunk
            else:
                # If not merging, the current buffer is a complete semantic chunk
                final_chunks.append(current_chunk_buffer)
                # Start a new buffer with the next chunk
                current_chunk_buffer = next_chunk

        # 3. Add the last remaining chunk buffer to the final list
        if current_chunk_buffer:
            final_chunks.append(current_chunk_buffer)

        return final_chunks
