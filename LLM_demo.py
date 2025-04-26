import json
import logging
from typing import Dict, List, Union, Generator, Any

from langchain_core.messages import SystemMessage, HumanMessage
import RAG
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

class RAG_LLM:
    """大模型交互类，支持意图识别和知识库查询"""
    
    
    #初始化大模型
    def __init__(self, llm_name="qwen2.5:7b-instruct"):
        try:
            self.llm = Ollama(model=llm_name, temperature=0.5)
            self.memory = ConversationBufferMemory()
            self.rag = RAG.RAG()  # 初始化知识库检索
            self.vector_store = self.rag.load_vector_store()
            logging.info(f"成功初始化模型: {llm_name}")
        except Exception as e:
            logging.error(f"初始化模型失败: {e}")
            raise
    # 流式输出
    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        return self.llm.stream(prompt)
    # 构建意图判断的prompt
    def get_intent_prompt(self, question: str) -> str:

        return f"""
        你需要对用户的问题进行意图分类，请回答"true"或"false"。
        - 如果用户询问菜谱推荐、食材搭配、烹饪方法或任何与烹饪和食物相关的问题，回答"true"
        - 如果用户的问题是日常闲聊、与烹饪无关的知识问答或其他类型的问题，回答"false"
        
        只回答"true"或"false"，不要解释。
        
        用户问题: {question}
        """
    
    
    
    # 构建日常聊天的prompt
    def get_chat_prompt(self, question: str) -> str:

        return f"""
        你是一位友好的AI助手"智小厨"，虽然你擅长烹饪领域的问题，但现在用户的问题与烹饪无关。
        请以温暖、专业的态度回答用户的问题。保持回答简洁、信息丰富且有帮助。不要透露你的模型相关信息。
        使用markdown格式输出哦
        
        用户问题: {question}
        """
    # 判断用户问题是否与烹饪相关
    def is_cooking_intent(self, question: str) -> bool:

        try:
            prompt = self.get_intent_prompt(question)
            response = self.llm.invoke(prompt).strip().lower()
            logging.info(f"意图判断结果: {response} (问题: {question[:30]}...)")
            return response == "true"
        except Exception as e:
            logging.error(f"意图判断出错: {e}")
            # 默认为非烹饪相关，避免不恰当地提供菜谱信息
            return False
    

    # 从知识库中检索相关菜谱信息
    def retrieve_recipes(self, question: str, top_k: int = 3) -> str:

        try:
            docs = self.rag.retrieve(self.vector_store, question, top_k)
            if not docs:
                return "未找到相关菜谱信息"
            
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            logging.error(f"知识库检索失败: {e}")
            return "知识库检索过程中出现错误"
            


    def chat(self, question: str, stream: bool = False):
        if not question.strip():
            return False, "请输入您的问题"

        is_cooking = self.is_cooking_intent(question)

        if is_cooking:
            context = self.retrieve_recipes(question)
            prompt = self.get_recipe_prompt(question, context)
        else:
            prompt = self.get_chat_prompt(question)

        if stream:
            return is_cooking, self.stream_response(prompt)
        else:
            response = self.llm.invoke(prompt)
            if is_cooking:
                return is_cooking,response
            return is_cooking, response
    



# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    