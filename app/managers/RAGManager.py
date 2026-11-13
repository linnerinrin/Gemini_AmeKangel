# """
#架构作用：为工具和会话提供知识/对话检索，召回后用交叉编码器重排
#数据流：本地文本或聊天轮→ Document → FAISS →相似度召回→重排→提示词文本
#生命周期：由managers包导入时创建；模型与索引在进程内共享，不持久化到数据库
# """
"""
managers/RAGManager.py检索增强生成管理器
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
    # """
    #同步加载embedding与cross-encoder；前者向量化，后者对候选逐对打分
    # """
    #阶段0会话运行时接入：每会话一个RAGManager实例（context索引按档隔离），
    #但编码器模型很重，进程内只加载一次，全实例共享同一份权重对象
    _shared_bi_encoder = None
    _shared_cross_encoder = None

    def __init__(self) ->None:
        #读
        super().__init__()
        self.recall_num=config.RAG_RECALL_SIZE
        self.rerank_num=config.RAG_RERANK_SIZE
        #单编码器用于recall（类级缓存，二次构造直接复用）
        if RAGManager._shared_bi_encoder is None:
            RAGManager._shared_bi_encoder=HuggingFaceEmbeddings(model_name=config.bi_encoder_dir,model_kwargs={'device':config.RAG_DEVICE})
        self.bi_encoder=RAGManager._shared_bi_encoder
        #双编码器用于rerank（同上，共享权重）
        if RAGManager._shared_cross_encoder is None:
            RAGManager._shared_cross_encoder=CrossEncoder(model_name=config.cross_encoder_dir).to(config.RAG_DEVICE)
        self.cross_encoder=RAGManager._shared_cross_encoder

        #当前场景轴由load_mode显式落定；未加载前为None，避免update_context读不到属性
        self.current_mode = None
        #context索引按存档分区：文档metadata打session_id标签，召回按当前绑定过滤
        self.current_session_id = None
        #每个存档已入库文档id（add_documents返回值），供clear_context按档删除
        self._context_doc_ids = {}

        #知识库与向量库
        self.knowledge =config.knowledge_prompt[config.INITIAL_MODE]
        self.context = []

        self.vector_store={"knowledge":None,"context":None}

        self.read_knowledge()

    def bind_session_id(self,session_id):
        #换绑存档只改过滤口径，不动索引；各档聊天互不召回
        self.current_session_id = session_id

    # """
    #场景变化只替换knowledge索引，context索引保留
    # """
    def load_mode(self,mode:str):
        self.current_mode = mode
        self.knowledge = config.knowledge_prompt[mode]
        self.read_knowledge()

    #读知识库初始化向量库
    # """
    #按非空行切文档并完整重建知识向量库；文件IO与embedding都在当前线程执行
    # """
    def read_knowledge(self)->None:
        chunks = []
        with open(str(self.knowledge),"r",encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                if line.strip(): chunks.append(Document(line))
        self.vector_store["knowledge"] = FAISS.from_documents(chunks, self.bi_encoder)

    # """
    #将完整问答编码成文档；空库初始化，已有库追加，没有按会话分区或删除旧会话
    # """
    def update_context(self,context:list)->None:
        # load_mode之前current_mode为None，标注退回默认场景而不是抛AttributeError
        mode_label = self.current_mode if self.current_mode is not None else config.INITIAL_MODE
        #未绑定存档不入context索引，避免无档聊天串进别的存档
        if not self.current_session_id:
            return
        #session_id只放metadata，不进正文（召回原文会拼进提示词，不能把内部标识漏给LLM）
        chunks=[Document(
            f"User:{line[0]['content']}\n{mode_label}:{line[1]['content']}",
            metadata={"session_id":self.current_session_id},
        ) for line in context]
        if self.vector_store["context"] is None:
            self.vector_store["context"]=FAISS.from_documents(chunks,self.bi_encoder)
            #首轮建库时全量文档即本存档文档
            self._context_doc_ids[self.current_session_id]=list(self.vector_store["context"].index_to_docstore_id.values())
        else:
            ids=self.vector_store["context"].add_documents(chunks)
            self._context_doc_ids.setdefault(self.current_session_id,[]).extend(ids)

    #清空对话context索引：指定session_id只删该存档文档，缺省全清（防旧聊天跨会话检索）
    def clear_context(self,session_id=None)->None:
        store=self.vector_store["context"]
        if store is None:
            self._context_doc_ids={}
            return
        if session_id is None:
            self.vector_store["context"]=None
            self._context_doc_ids={}
            return
        ids=self._context_doc_ids.pop(session_id,[])
        if ids:
            try:
                store.delete(ids)
            except Exception as e:
                #FAISS删除失败会留半挂索引；内容本就运行期易失，全清重建更安全
                print(f"清空存档上下文失败，重建context索引:{e}")
                self.vector_store["context"]=None
                self._context_doc_ids={}

    #召回单编码器粗筛效率高召回率低
    # """
    #根据caller选knowledge/context，空索引返回空候选；向量召回仅产生原始文本列表
    # """
    def recall(self,query:str,caller:str)->list:
        #计算与查询相似度最高的几个向量返回文档
        store = self.vector_store.get(caller)
        if store is None:
            return []
        #context索引按存档过滤：只召回当前绑定会话的聊天，老文档无metadata视为非本会话
        search_filter = None
        if caller == "context":
            if not self.current_session_id:
                return []
            search_filter = {"session_id": self.current_session_id}
        results = store.similarity_search(query, k=self.recall_num, filter=search_filter)
        documents = [rst.page_content for rst in results]

        return documents

    #重排双编码器精排效率高召回率高
    # """
    #对候选以查询-文本对打分，按corpus_id回取文本，返回prompts/value字典列表
    #重排不构成来源可信性检查，文本仍可能包含来自用户或工具的指令性内容
    # """
    def rerank(self,query:str,documents:list)->list:
        final_result=[]
        #用双编码器计算召回结果与相似度最高的几个向量返回文档
        rerst = self.cross_encoder.rank(
            query,
            documents,
            top_k=self.rerank_num
        )
        for item in rerst:
            id, score = item["corpus_id"], item["score"]
            final_result.append({"prompts":documents[id],"value":score})
        return final_result

    #检索异步召回+重排
    # """
    #召回和重排分别在线程池执行，await后拼接提示文本；线程卸载只覆盖这两步
    #建库/追加索引仍同步执行，共享模型与索引没有按会话独立实例
    # """
    async def retrieve(self, query: str, caller: str = "knowledge") -> str:
        rag_prompt = ""
        recall_result = await asyncio.to_thread(self.recall, query, caller)
        final_result = await asyncio.to_thread(self.rerank, query, recall_result)
        for item in final_result:
            rag_prompt += f"{item['prompts']}\n"
        return rag_prompt
