from typing import List, Dict

from utils import DialogManager
from sentence_transformers import util
from app.builders import VectorDBBuilder, KVCacheBuilder

class RAGCAGProcessor:
    def __init__(self, vectordb: VectorDBBuilder, kvcache: KVCacheBuilder, tokenizer, model):
        self.vectordb = vectordb
        self.kvcache = kvcache
        self.tokenizer = tokenizer
        self.model = model
        self.dialog = DialogManager()

    def match_cache(self, query: str):
        for cached in self.kvcache.cache_store.keys():
            if query.startswith(cached):  # prefix match
                return cached
            sim = util.cos_sim(self.vectordb.embedder.encode(query, convert_to_tensor=True),
                               self.vectordb.embedder.encode(cached, convert_to_tensor=True))
            if sim.item() > 0.8:
                return cached
        return None

    def generate_response(self, query: str):
        matched = self.match_cache(query)
        if matched:
            VectorDBBuilder.clean_up()
            input_ids = self.tokenizer(query, return_tensors="pt").input_ids.to(self.model.device)
            outputs = self.model(input_ids=input_ids, past_key_values=self.kvcache.get_cache(matched))
            response = self.tokenizer.decode(outputs.logits.argmax(-1)[0], skip_special_tokens=True)
        else:
            retrieved = self.vectordb.search_db(query)
            context = "\n".join(retrieved)
            full_prompt = context + "\n" + query
            input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.model.device)
            outputs = self.model.generate(input_ids)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # create a new cache
            self.kvcache.build_kv_cache([full_prompt])

        self.dialog.add_turn(query, response)
        return response

    def get_conversation_cache(self) -> List[Dict[str, str]]:
        return self.dialog.get_history()