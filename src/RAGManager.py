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
        self.recall_num=config['rag_recall_size']
        self.rerank_num=config['rag_rerank_size']
        self.bi_encoder=HuggingFaceEmbeddings(model_name=str(BASE_DIR / config['bi_encoder_paths']),model_kwargs={'device':'cuda'})
        self.cross_encoder=CrossEncoder(model_name=str(BASE_DIR / config['cross_encoder_paths'])).to("cuda")
        self.knowledge = None
        self.vector_store = None

    def read(self,knowledge:str)->None:
        chunks = []
        with open(str(BASE_DIR / knowledge),"r",encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                if line.strip(): chunks.append(Document(line))
        self.knowledge=chunks
        self.vector_store = FAISS.from_documents(self.knowledge, self.bi_encoder)

    def recall(self,query:str)->list:
        results = self.vector_store.similarity_search(query, k=self.recall_num)
        documents = [rst.page_content for rst in results]

        return documents

    def rerank(self,query:str,documents:list)->list:
        final_result=[]
        rerst = self.cross_encoder.rank(
            query,
            documents,
            top_k=self.rerank_num
        )
        for item in rerst:
            id, score = item["corpus_id"], item["score"]
            final_result.append({"text":documents[id],"value":score})
        return final_result

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

