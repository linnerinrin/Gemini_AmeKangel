"""
普通检索类       --包含read recall rerank retrieve
RAGManager      --实现了IRAGjiekou
    read        --读取知识库
    recall      --文档召回
    rerank      --文档重排
    retrieve    --文档检索 综合了召回和重排
"""

import asyncio
from pathlib import Path
from interfaces.IRAG import IRAG
from langchain_community.vectorstores import FAISS
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
class RAGManager(IRAG):
    def __init__(self,config:dict)->None:
        #读
        self.recall_num=config['rag_recall_size']
        self.rerank_num=config['rag_rerank_size']

        #加载模型
            #hf模型是参数内开cuda
            #transformers模型是外部to cuda
            #unsloth自带cuda(硬性要求cuda版pytorch)
        #单编码器 用于recall
        self.bi_encoder=HuggingFaceEmbeddings(model_name=str(BASE_DIR / config['bi_encoder_paths']),model_kwargs={'device':'cuda'})
        #双编码器 用于rerank
        self.cross_encoder=CrossEncoder(model_name=str(BASE_DIR / config['cross_encoder_paths'])).to("cuda")

        #知识库与向量库
        self.knowledge = None
        self.vector_store = None

    #读知识库 初始化向量库
    def read(self,knowledge:str)->None:
        chunks = []
        #读知识库
        with open(str(BASE_DIR / knowledge),"r",encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                if line.strip(): chunks.append(Document(line))
        self.knowledge=chunks
        #初始化faiss
        self.vector_store = FAISS.from_documents(self.knowledge, self.bi_encoder)

    #召回 单编码器粗筛 效率高召回率低
    def recall(self,query:str)->list:
        #计算与查询相似度最高的几个向量 返回文档
        results = self.vector_store.similarity_search(query, k=self.recall_num)
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
            final_result.append({"text":documents[id],"value":score})
        return final_result

    #检索 异步召回+重排
    async def retrieve(self,query:str,knowledge:str)->str:
        self.read(knowledge)
        rag_prompt=""
        task1 = asyncio.to_thread(self.recall, query)
        recall_result=await task1
        task2 = asyncio.to_thread(self.rerank, query, recall_result)
        final_result=await task2
        for item in final_result:
            rag_prompt+=f"{item['text']}\n"
        return rag_prompt

