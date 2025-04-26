import json
import logging
from typing import Dict, List, Union, Generator, Any

from langchain_community.llms import Ollama

class llm_model:
    
    #初始化大模型
    def __init__(self, llm_name="qwen2.5:7b-instruct"):
        try:
            self.llm = Ollama(model=llm_name, temperature=0.5)
            logging.info(f"成功初始化模型: {llm_name}")
        except Exception as e:
            logging.error(f"初始化模型失败: {e}")
            raise e

        # 处理用户问题，返回回答

        
    
    # def prompt_template(user_input, context="暂无", task_description="暂无"):
    #     prompt = f"""
    # 你是一个智能老人关怀语音助手，名字叫“小爱”。你的目标是陪伴和帮助老年人，语气要温暖、亲切、耐心，像一个贴心的孙子/孙女一样。使用简洁、易懂的中文，避免复杂术语，语句要符合老年人的语言习惯。

    # 当前任务：{task_description}
    # 用户输入：{user_input}
    # 上下文：{context}

    # 根据用户输入和上下文，生成一段适合与老年人交互的语音回复，语气要自然、友好，语句长度适中（每句不超过20字），避免一次性说太多。如果需要，可以主动提出建议或询问需求。
    # """
    #     return prompt
    def prompt_template(user_input,  recipe_data="暂无",context="暂无"):
        prompt = f"""
    你是一个智能做菜助手，名字叫智小厨。你的目标是根据用户当前做的菜，提供相关的做菜建议和帮助。语气要温暖、亲切、耐心。


    当前菜品数据：{recipe_data}
    用户输入：{user_input}
    上下文：{context}
    回答用户问题。
"""
        return prompt    


    def chat(self, question: str):
        prompt = self.prompt_template(question)
        try:
            response = self.llm.stream(prompt)
            for chunk in response:
                yield chunk
        except Exception as e:
            return response
        except Exception as e:
            logging.error(f"生成回答失败: {e}")
            return None
        
        
    



# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    