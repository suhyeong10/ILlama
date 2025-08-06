import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from illama.illama import ILlama


async def main():
    # 1. Initialize ILlama
    print('Initialize ILlama...')
    illama = ILlama(
        model_name='Codingchild/ILlama-8b-LoRA',
        retriever_name='Codingchild/medical-bge-large-en-v1.5',
        reranker_name='Codingchild/medical-bge-reranker-large',
    )

    # 2. Prepare ILlama
    print('Prepare ILlama...')
    chain = illama.prepare_illama(
        generation_type='w_rag'
    )

    # 3. Inference
    print('Inference...')
    response = await illama.inference(
        chain=chain,
        query='I have a headache and a fever. What should I do?'
    )

    print(response)

if __name__ == '__main__':
    asyncio.run(main())