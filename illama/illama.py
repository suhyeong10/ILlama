import os
import sys
from typing import Optional

from langchain_core.runnables import Runnable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from illama.retriever import PDFIngestor
from illama.chain import get_llm, get_chain
from illama.prompts import PromptGenerator


class ILlama:
    """
    This class is used to create a chain of the model.
    The chain is used to generate the response for the question.
    The chain is created with the model, the retriever, and the prompt.
    The retriever is used to retrieve the most relevant documents for the question.

    Args:
        model_name: str, the name of the model to use
        max_new_tokens: int, the maximum number of tokens to generate
        do_sample: bool, whether to sample from the model
        num_beams: int, the number of beams to use
        repetition_penalty: float, the repetition penalty
        length_penalty: float, the length penalty  
        early_stopping: bool, whether to early stop the generation
        no_repeat_ngram_size: int, the size of the ngram to avoid repeating
        temperature: float, the temperature of the model
        top_p: float, the top p of the model
        top_k: int, the top k of the model
        llm_dtype: str, the dtype of the model
        retriever_dtype: str, the dtype of the retriever
        retriever_name: str, the name of the retriever to use
        reranker_name: str, the name of the reranker to use
        pdf_path: str, the path to the PDF files
        text_save_path: str, the path to save the text
        vector_store_path: str, the path to save the vector store
        retrieve_k: int, the number of documents to retrieve
        retrieve_n: int, the number of documents to retrieve
        use_rag: bool, whether to use the RAG

    Returns:
        ILlama, the instance of the class
    """
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        num_beams: int = 4,
        repetition_penalty: float = 1.18,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 4,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        llm_dtype: str = "bfloat16",
        retriever_dtype: str = "float32",
        retriever_name: str = None,
        reranker_name: str = None,
        pdf_path: str = 'datasets/database/kg_data',
        text_save_path: str = 'datasets/database',
        vector_store_path: str = 'datasets/database',
        retrieve_k: int = 50,
        retrieve_n: int = 10,
        use_rag: bool = True
    ):


        self.prompt = None
        self.retriever = None
        self.llm, self.model, self.tokenizer = get_llm(
            model_name=model_name, 
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            dtype=llm_dtype
        )

        if use_rag:
            self.db = PDFIngestor(
                model_name=model_name,
                tokenizer=self.tokenizer,
                torch_dtype=retriever_dtype,
                pdf_path=pdf_path,
                text_save_path=text_save_path,
                vector_store_path=vector_store_path
            )

            self.retriever = self.db.get_compressor(reranker_name=reranker_name, top_n=retrieve_n)

    def prepare_illama(self, generation_type: str = 'w_rag', prompt: Optional[str] = None):
        self.prompt_generator = PromptGenerator(gen_type=generation_type, prompt=prompt)
        self.prompt = self.prompt_generator.generate_prompt()

        chain = get_chain(
            prompt=self.prompt,
            retriever=self.retriever,
            llm=self.llm
        )

        return chain

    async def inference(self, chain: Runnable, query: str):
        response = await chain.ainvoke(query)
        
        return response
