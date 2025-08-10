import os
import shutil
from typing import List

import torch
import faiss
from pandas.io.clipboard import paste
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "./cache"

PROMPT_TEMPLATE = """
Please answer the question based only on the following context:

{context}

---

Please Answer the question based on the above context: {question}
"""

class VectorDBBuilder:
    def __init__(self, embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.index = None
        self.text_chunks = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        self.metadata = None

    def _split_text(self, documents: list):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        document = chunks[10]
        print(document.page_content)
        print(document.metadata)

        return chunks

    def _save_to_db(self, chunks: list):
        # Clear out the database first.
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        # Create a new DB from the documents.
        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

    def build_vector_index(self, documents: List[str]):
        # Simple chunking by sentences
        #self.text_chunks = [t.strip() for text in texts for t in text.split("。") if t.strip()]
        # embeddings = self.embedder.encode(self.text_chunks, convert_to_tensor=True, show_progress_bar=True)
        #
        # self.index = faiss.IndexFlatL2(embeddings.shape[1])
        # self.index.add(embeddings.cpu().detach().numpy())
        #
        text_chunks = self._text_splitter.split_documents(documents)

        self._save_to_db(text_chunks)

    def search_db(self, query: str, top_k: int = 5):
        # query_embedding = self.embedder.encode([query], convert_to_tensor=True)
        # D, I = self.index.search(query_embedding.cpu().detach().numpy(), top_k)
        # return [self.text_chunks[i] for i in I[0]]

        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        results = db.similarity_search_with_relevance_scores(query, k=top_k)
        if len(results) == 0 or results[0][1] < 0.7:
            print(f"Unable to find matching results.")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)
        print(prompt)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)

class KVCacheBuilder:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.device = model.model.embed_tokens.weight.device
        #self.cache_store = {}
        self.cache_store = None
        self.origin_len = 0


    def build_kv_cache(self, prompts: List[str]):
        cache = DynamicCache()
        # for prompt in prompts:
        #     input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        #     with torch.no_grad():
        #         _ = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
        #     self.cache_store[prompt] = cache
        #return cache

        input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids.to(self.device)
        self.origin_len = input_ids.shape[-1]
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
        self.cache_store = cache


    def clean_up(self, cache: DynamicCache, origin_len: int):
        for i in range(len(cache.key_cache)):
            self.cache_store.key_cache[i] = self.cache_store.key_cache[i][:, :, :self.origin_len , :]
            self.cache_store.value_cache[i] = self.cache_store.value_cache[i][:, :, :self.origin_len , :]

    def get_cache(self, context: str):
        return self.cache_store.get(context)

    def has_cache(self, context: str) -> bool:
        return context in self.cache_store