import os
import sys
from typing import Optional

from langchain_core.runnables import Runnable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from illama.retriever import PDFIngestor
from illama.chain import get_llm, get_chain
from illama.prompts import PromptGenerator


class ILlama:
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
