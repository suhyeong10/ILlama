import os
import sys
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

from langchain.retrievers import ContextualCompressionRetriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable

from langchain_huggingface.llms import HuggingFacePipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from illama.retriever import format_docs


def get_llm(
    model_name: str, 
    max_new_tokens: int,
    do_sample: bool,
    num_beams: int,
    repetition_penalty: float,
    length_penalty: float,
    early_stopping: bool,
    no_repeat_ngram_size: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    dtype: str = "bfloat16",
) -> HuggingFacePipeline:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype
    )

    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        no_repeat_ngram_size=no_repeat_ngram_size,
        temperature=temperature,
        top_p=top_p,      
        top_k=top_k
    )

    return HuggingFacePipeline(pipeline=generation_pipeline), model, tokenizer


def get_chain(
    prompt: ChatPromptTemplate,
    retriever: Optional[ContextualCompressionRetriever] = None,
    llm: HuggingFacePipeline = None,
) -> Runnable:

    if retriever is not None:
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    return chain