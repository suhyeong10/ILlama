# 🩺 ILlama
> Leveraging Knowledge Graph-Enhanced LLMs for Context-Aware Medical Consultation (EMNLP 2025)
> A novel RAG framework leveraging structured medical knowledge via subgraphs for context-aware, hallucination-reduced medical consultations.

---

## 🗝️ Key feature
- ✅ **Enhanced Factual Reliability**: Integrates explicit causal medical relationships to significantly reduce inaccuracies.
- 🔎 **Precise Contextual Retrieval**: Leverages transformed knowledge sub-units and vector search for accurate information integration.
- 💉 **Superior Clinical Utility**: Achieves state-of-the-art performance for reliable and practical medical guidance.

<p align="center">
  <img src="./assets/ILlama.jpg" width="70%" alt="Framework Overview">
</p>

<p align="center">
  <img src="./assets/data_gen.jpg" width="70%" alt="Data Generation Pipeline">
</p>

---

## 📖 Usage
### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Quick Start Example
```python
import asycnio
from illama.illama import ILlama

illama = ILlama(
        model_name='Codingchild/ILlama-8b-LoRA',
        retriever_name='Codingchild/medical-bge-large-en-v1.5',
        reranker_name='Codingchild/medical-bge-reranker-large',
    )

asycn def main():
    # prepare illama model & prompt
    chain = illama.prepare_illama(
        generation_type='w_rag'
    )

    # generate response for user's query
    response = await illama.inference(
        chain=chain,
        query='I have a headache and a fever. What should I do?'
    )

    print(response)

asyncio.run(main())
```

## 📜 Citation
If you find this work helpful, please consider citing us:
```bibtex
@inproceedings{anonymous2025leveraging,
    title={Leveraging Knowledge Graph-Enhanced {LLM}s for Context-Aware Medical Consultation},
    author={Anonymous},
    booktitle={Submitted to ACL Rolling Review - May 2025},
    year={2025},
    url={https://openreview.net/forum?id=2oOy0FvWcB},
    note={under review}
}
```

## 📬 Contact
Email: pshpulip22@catholic.ac.kr
