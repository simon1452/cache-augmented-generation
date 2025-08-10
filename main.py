import os
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.doc_preprocessor import DocPreprocessor
from app.builders import VectorDBBuilder, KVCacheBuilder
from app.processor import RAGCAGProcessor

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# initial model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN)

# handle documents
doc_processor = DocPreprocessor()
documents = doc_processor.process_documents(["docs/MG7 车主用户手册_20230609.pdf", "docs/data.xlsx"])

# create Vector DB
vectordb = VectorDBBuilder()
vectordb.build_vector_index(documents)

# create KV Cache
kvcache = KVCacheBuilder(tokenizer, model)
kvcache.build_kv_cache(documents)

ragcag = RAGCAGProcessor(vectordb, kvcache, tokenizer, model)


def generate_response(query: str):
    response = ragcag.generate_response(query)
    print(f"User: {query}")
    print(f"AI  : {response}")
    return response


if __name__ == "__main__":
    while True:
        q = input("enter your questions or type exit/quit：")
        if q.lower() in ["exit", "quit"]:
            break
        generate_response(q)
