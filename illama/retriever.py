import os
import pickle
from tqdm import tqdm

import torch

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_transformers import LongContextReorder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_huggingface import HuggingFaceEmbeddings

reordering = LongContextReorder()

class PDFIngestor:
    """
    This class is used to ingest the PDF files and create a vector store.
    The vector store is used to retrieve the most relevant documents for the question.
    The documents are split into chunks and then embedded.
    The chunks are then stored in a FAISS vector store.
    """
    def __init__(
            self,
            model_name,
            tokenizer,
            torch_dtype: str = "float32",
            pdf_path: str = 'datasets/database/kg_data',
            text_save_path: str = 'datasets/database',
            vector_store_path: str = None
        ):

        self.torch_dtype = torch_dtype
        self.vector_store_path = vector_store_path
        self.pdf_path = pdf_path
        self.text_save_path = text_save_path

        if not os.path.isfile(self.text_save_path + '/kg_docs.pkl'):
            self.docs_list = self.get_docs()

            self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=tokenizer,
                chunk_size=128,
                chunk_overlap=0,
            )

            doc_splits = self.text_splitter.split_documents(self.docs_list)

            with open(f'{self.text_save_path}/kg_docs.pkl', 'wb') as f:
                pickle.dump(doc_splits, f)
        else:
            with open(f'{self.text_save_path}/kg_docs.pkl', 'rb') as f:
                doc_splits = pickle.load(f)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
                "model_kwargs": {
                    'torch_dtype': self.torch_dtype
                }
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        if os.path.exists(self.vector_store_path) and self.vector_store_path is not None:
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = FAISS.from_documents(
                documents=doc_splits,
                embedding=self.embeddings,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )

            self.vector_store.save_local(self.vector_store_path)

    def get_docs(self):
        docs_list = list()

        if os.path.isdir(self.pdf_path):
            pdf_files = [file_name for file_name in os.listdir(self.pdf_path) if file_name.endswith(".pdf")]
            for file_name in tqdm(pdf_files, desc="Loading PDF files", unit="file", ncols=150):
                pdf_file_path = os.path.join(self.pdf_path, file_name)
                docs_list.append(PyPDFLoader(pdf_file_path).load())
                
        documents_list = [item for sublist in docs_list for item in sublist]
        
        return documents_list
    
    def get_retriever(self, top_k=50):
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})

    def get_compressor(self, reranker_name, top_n=10):
        retriever = self.get_retriever()

        cross_encoder = HuggingFaceCrossEncoder(
            model_name=reranker_name,
            model_kwargs={
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
                "automodel_args": {
                    "torch_dtype": self.torch_dtype
                }
            }
        )

        reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=retriever
        )

        return compression_retriever


def format_docs(docs):
    reordered_docs = reordering.transform_documents(docs)

    return "\n\n".join([d.page_content for d in reordered_docs])
