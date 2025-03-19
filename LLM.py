from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import RAG
# 初始化本地模型
class RAG_LLM():
    def __init__(self):
        self.llm=Ollama(model="qwq:latest", temperature=0.5)
    
    # 意图判断函数
    def is_intent_with_llm(self,question: str) -> bool:
 
        # 构造提示词，要求模型进行二分类判断
        template_messages = [
           SystemMessage(content="""
        现在你是一个意图判断助手，你需要根据用户的提问判断用户的意图，并以字典返回。
        简单来说，你现在需要做出二分类判断并直接输出简单的判断结果，如true和false
        当用户提问需要进行菜谱推荐时，或回答与菜谱知识想关的问题时，回答true
        当用户的提问你不需要借助菜谱知识时，回答false

        如'用户提问：我有牛肉和土豆，可以做什么菜？ 你需要回答：true'
        如'用户提问：今天天气怎么样？ 你需要回答：false'
           """),
           HumanMessage(content="用户提问：{question}")
        ]
        formatted_messages = []
        for msg in template_messages:
            formatted_content = msg.content.format(question=question)
            if isinstance(msg, SystemMessage):
                formatted_messages.append(SystemMessage(content=formatted_content))
            elif isinstance(msg, HumanMessage):
                formatted_messages.append(HumanMessage(content=formatted_content))

        # 拼接成完整的 prompt 字符串（也可以直接传递消息列表给部分链）
        full_prompt = "\n".join([msg.content for msg in formatted_messages])
        res=self.llm.invoke(full_prompt)
        if res=='false':
            return False
        else :
            return True
        

        #
    def llm_ask(self,question: str) -> str:
        flag=self.is_intent_with_llm(question)
        
        if flag:
            template_messages = [
            SystemMessage(content="""你是一个做菜小助手，名字叫智小厨，可以回答用户的问题。
    - 你需要基于以下结构化菜谱知识推荐合适1-3个菜谱，并给出推荐理由。
    菜谱知识：{query}
    你可以润色知识库的内容，但不能改变其含义。
    你的回答需要以 JSON 格式字符串输出，包含以下字段：
        - "recipe": 菜谱数据（从知识库中提取出菜品id返回）
        - "reason": 推荐理由（字符串）
    """),
            HumanMessage(content="用户提问：{question}")
            ]
            # 从知识库检索
            rag = RAG.RAG()
            vector_store = rag.load_vector_store()
            query_old = rag.retrieve(vector_store, question, 5)
            query = "\n".join([doc.page_content for doc in query_old]) if query_old else "暂无相关菜谱知识"
            formatted_messages = []
            for msg in template_messages:
                formatted_content = msg.content.format(query=query, question=question)
                if isinstance(msg, SystemMessage):
                    formatted_messages.append(SystemMessage(content=formatted_content))
                elif isinstance(msg, HumanMessage):
                    formatted_messages.append(HumanMessage(content=formatted_content))
            full_prompt = "\n".join([msg.content for msg in formatted_messages])
            










