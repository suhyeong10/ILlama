from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = {}
SYSTEM_PROMPT['w_rag'] = """You are a medical assistant specializing in providing expert consultations for medical inquiries.
Your role is to deliver accurate, user-friendly medical information, clarify symptoms, explain potential
medical conditions, and recommend next steps with empathy and professionalism. 
When formulating your response, to ensure clarity and accuracy, provide a user-friendly answer in your response.

Please make sure to base your answer primarily on the provided context retrieved from the external knowledge sources.
Use only the information found in the context to support your response, and do not include any content that is not grounded in it.
If the context does not contain enough information to answer confidently, politely acknowledge this and recommend seeking advice from a medical professional.
"""
SYSTEM_PROMPT['wo_rag'] = """You are a medical assistant specializing in providing expert consultations for medical inquiries.
Your role is to deliver accurate, user-friendly medical information, clarify symptoms, explain potential
medical conditions, and recommend next steps with empathy and professionalism. 
When formulating your response, to ensure clarity and accuracy, provide a user-friendly answer in your response.
"""
SYSTEM_PROMPT['custom'] = """
"""


class PromptGenerator:
    """
    This class is used to generate prompts for the model.
    The prompts are generated based on the type of prompt to generate.
    The type of prompt can be 'w_rag' or 'wo_rag' or 'custom'.
    If the type is 'w_rag', the prompt will be generated with the context and the question.
    If the type is 'wo_rag', the prompt will be generated with the question.
    If the type is 'custom', the prompt will be generated with the custom prompt.

    Args:
        gen_type: str, the type of prompt to generate
        prompt: str, the prompt to generate

    Returns:
        ChatPromptTemplate, the prompt template
    """
    def __init__(self, gen_type: str = 'w_rag', prompt: str = None):
        self.gen_type = gen_type
        if gen_type == 'w_rag':
            self.system_prompt = SYSTEM_PROMPT['w_rag']
        elif gen_type == 'wo_rag':
            self.system_prompt = SYSTEM_PROMPT['wo_rag']
        else:
            self.system_prompt = prompt
        
    def generate_prompt(self):
        if self.gen_type == 'w_rag':
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("user", "\n### Context\n{context}\n\n### Question\n{question}")
            ])
        elif self.gen_type == 'wo_rag':
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("user", "\n### Question\n{question}")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("user", "\n### Question\n{question}")
            ])
        return prompt
    