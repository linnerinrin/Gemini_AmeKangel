import pytest
import sys
import types

from conftest import load_source_module


class FakeFAISS:
    @staticmethod
    def from_documents(chunks, encoder):
        return None


class FakeCrossEncoderModel:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self


class FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        pass


class FakeDocument:
    def __init__(self, page_content):
        self.page_content = page_content


langchain_community = types.ModuleType("langchain_community")
vectorstores = types.ModuleType("langchain_community.vectorstores")
vectorstores.FAISS = FakeFAISS
sentence_transformers = types.ModuleType("sentence_transformers")
cross_encoder = types.ModuleType("sentence_transformers.cross_encoder")
cross_encoder.CrossEncoder = FakeCrossEncoderModel
langchain_huggingface = types.ModuleType("langchain_huggingface")
langchain_huggingface.HuggingFaceEmbeddings = FakeEmbeddings
langchain_core = types.ModuleType("langchain_core")
documents = types.ModuleType("langchain_core.documents")
documents.Document = FakeDocument
sys.modules["langchain_community"] = langchain_community
sys.modules["langchain_community.vectorstores"] = vectorstores
sys.modules["sentence_transformers"] = sentence_transformers
sys.modules["sentence_transformers.cross_encoder"] = cross_encoder
sys.modules["langchain_huggingface"] = langchain_huggingface
sys.modules["langchain_core"] = langchain_core
sys.modules["langchain_core.documents"] = documents

rag_module = load_source_module("rag_manager_under_test", "app/managers/RAGManager.py")


class FakeStore:
    def similarity_search(self, query, k):
        return [type("Doc", (), {"page_content": f"{query}:{k}"})()]

    def add_documents(self, docs):
        self.docs = docs


class FakeCrossEncoder:
    def rank(self, query, documents, top_k):
        return [{"corpus_id": 0, "score": 0.9}]


def test_recall_returns_document_content():
    mgr = rag_module.RAGManager.__new__(rag_module.RAGManager)
    mgr.recall_num = 1
    mgr.vector_store = {"knowledge": FakeStore()}

    assert mgr.recall("q", "knowledge") == ["q:1"]


def test_recall_missing_store_returns_empty_list():
    mgr = rag_module.RAGManager.__new__(rag_module.RAGManager)
    mgr.vector_store = {"context": None}

    assert mgr.recall("q", "context") == []


def test_rerank_maps_scores():
    mgr = rag_module.RAGManager.__new__(rag_module.RAGManager)
    mgr.rerank_num = 1
    mgr.cross_encoder = FakeCrossEncoder()

    assert mgr.rerank("q", ["doc"]) == [{"prompts": "doc", "value": 0.9}]
