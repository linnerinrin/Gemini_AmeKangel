"""
managers/RAGManager.py 检索增强生成管理器
Abstract on IRAG
"""

import asyncio

from langchain_community.vectorstores import FAISS
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config.config import config
from app.interfaces import IRAG


class RAGManager(IRAG):
    def __init__(self) ->None:
        #读
        super().__init__()
        self.recall_num=config.RAG_RECALL_SIZE
        self.rerank_num=config.RAG_RERANK_SIZE
        #单编码器 用于recall
        self.bi_encoder=HuggingFaceEmbeddings(model_name=config.bi_encoder_dir,model_kwargs={'device':config.RAG_DEVICE})
        #双编码器 用于rerank
        self.cross_encoder=CrossEncoder(model_name=config.cross_encoder_dir).to(config.RAG_DEVICE)

        #知识库与向量库
        self.knowledge =config.knowledge_prompt[config.INITIAL_MODE]
        self.context = []

        self.vector_store={"knowledge":None,"context":None}

        self.read_knowledge()

    def load_mode(self,mode:str):
        self.current_mode = mode
        self.knowledge = config.knowledge_prompt[mode]
        self.read_knowledge()

    #读知识库 初始化向量库
    def read_knowledge(self)->None:
        chunks = []
        with open(str(self.knowledge),"r",encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                if line.strip(): chunks.append(Document(line))
        self.vector_store["knowledge"] = FAISS.from_documents(chunks, self.bi_encoder)

    def update_context(self,context:list)->None:
        chunks=[Document(f"User:{line[0]['content']}\n{self.current_mode}:{line[1]['content']}") for line in context]
        if self.vector_store["context"] is None:
            self.vector_store["context"]=FAISS.from_documents(chunks,self.bi_encoder)
        else:
            self.vector_store["context"].add_documents(chunks)

    #召回 单编码器粗筛 效率高召回率低
    def recall(self,query:str,caller:str)->list:
        #计算与查询相似度最高的几个向量 返回文档
        store = self.vector_store.get(caller)
        if store is None:
            return []
        results = store.similarity_search(query, k=self.recall_num)
        documents = [rst.page_content for rst in results]

        return documents

    #重排 双编码器精排 效率高召回率高
    def rerank(self,query:str,documents:list)->list:
        final_result=[]
        #用双编码器计算召回结果与相似度最高的几个向量 返回文档
        rerst = self.cross_encoder.rank(
            query,
            documents,
            top_k=self.rerank_num
        )
        for item in rerst:
            id, score = item["corpus_id"], item["score"]
            final_result.append({"prompts":documents[id],"value":score})
        return final_result

    #检索 异步召回+重排
    async def retrieve(self, query: str, caller: str = "knowledge") -> str:
        rag_prompt = ""
        recall_result = await asyncio.to_thread(self.recall, query, caller)
        final_result = await asyncio.to_thread(self.rerank, query, recall_result)
        for item in final_result:
            rag_prompt += f"{item['prompts']}\n"
        return rag_prompt

