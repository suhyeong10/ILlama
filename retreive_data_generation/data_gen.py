import os
import sys

from transformers import pipeline, set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retreive_data_generation.prompts import generate_prompt


set_seed(42)

class DataGenerator:
    """
    This class is used to generate data for the model.
    The data is generated based on the context and the question.
    The data is generated with the model, the generator, and the evaluator.

    Args:
        model_name: str, the name of the model to use
        output_path: str, the path to save the data
        max_attempts: int, the maximum number of attempts to generate the data
        max_new_tokens: int, the maximum number of tokens to generate
        device: str, the device to use

    Returns:
        DataGenerator, the instance of the class
    """
    def __init__(
        self, 
        model_name: str, 
        output_path: str,
        max_attempts: int,
        max_new_tokens: int,
        device: str,
    ):

        self.model_name = model_name
        self.output_path = output_path
        self.max_attempts = max_attempts
        self.max_new_tokens = max_new_tokens
        self.device = device

    def generator(self):
        return pipeline(
            "text-generation",
            model=self.model_name,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )

    def evaluator(self):
        return pipeline(
            "text-generation",
            model=self.model_name,
            device=self.device,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )

    def generate_query(
        self,
        context: str,
        llm: pipeline = None,
    ):
        prompt = generate_prompt(prompt_type='generate')
        prompt = prompt.format(context=context)

        output = llm(
            prompt,
            return_full_text=False,
            skip_special_tokens=True
        )

        return output.strip()

    def regenerate_query(
        self,
        query: str = None,
        context: str = None,
        patient_score: float = None,
        relevance_score: float = None,
        llm: pipeline = None,
    ):
        prompt = generate_prompt(prompt_type='regenerate')
        prompt = prompt.format(
            context=context,
            query=query,
            patient_score=patient_score,
            relevance_score=relevance_score
        )

        output = llm(
            prompt,
            return_full_text=False,
            skip_special_tokens=True
        )

        return output.strip()
    
    def evaluate_query(
        self,
        query: str,
        context: str,
        llm: pipeline = None,
    ):
        prompt = generate_prompt(prompt_type='evaluate')
        prompt = prompt.format(
            context=context,
            query=query
        )
        
        output = llm(
            prompt,
            return_full_text=False,
            skip_special_tokens=True
        )

        return output.strip()
    
    def generation_data(
        self,
        context: str,
        max_attempts: int = 5,
        generator: pipeline = None,
        evaluator: pipeline = None,
    ):
        query = self.generate_query(context=context, llm=generator)
        scores = self.evaluate_query(query, context, llm=evaluator)

        patient_score = scores["patient_score"]
        relevance_score = scores["relevance_score"]

        if patient_score >= 0.5 and relevance_score >= 0.5:
            return context, query, scores
        else:
            for _ in range(max_attempts):
                query = self.regenerate_query(
                    query=query,
                    context=context,
                    patient_score=patient_score,
                    relevance_score=relevance_score,
                    llm=generator
                )

                scores = self.evaluate_query(query, context, llm=evaluator)

                patient_score = scores["patient_score"]
                relevance_score = scores["relevance_score"]

                if patient_score >= 0.5 and relevance_score >= 0.5:
                    break

            return context, query, scores