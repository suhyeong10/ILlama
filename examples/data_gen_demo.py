import os
import sys
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieve_data_generation.data_gen import DataGenerator

def main():
    # 1. load context data
    print('Load context data...')
    with open('datasets/context.jsonl', 'r') as f:
        context_data = [json.loads(line) for line in f]

    context_list = [item['context'] for item in context_data]

    # 2. Initialize DataGenerator
    print('Initialize DataGenerator...')
    data_generator = DataGenerator(
        model_name='meta-llama/Llama-3.1-8B',
        output_path='datasets/queries.json',
        max_attempts=5,
        max_new_tokens=512,
        device='cuda:0'
    )

    # 3. Initialize generator and evaluator
    print('Initialize generator and evaluator...')
    generator = data_generator.generator()
    evaluator = data_generator.evaluator()

    # 4. Generate queries
    print('Generate queries...')
    for context in tqdm(context_list, desc='Generating queries'):
        context, query, _ = data_generator.generation_data(context, generator, evaluator)  

        output = {
            "context": context,
            "query": query,
        } 

        with open('datasets/retriever_data.jsonl', 'a') as f:
            json_output = json.dumps(output, indent=4)
            f.write(json_output)

if __name__ == "__main__":
    main()
