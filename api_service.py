import logging
import json
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import KG
import pandas as pd
import numpy as np
from pydantic import BaseModel

from LLM_demo import RAG_LLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from GNN import GNNRecommender



# 定义请求和响应模型
class RecommendRequest(BaseModel):
    user_features: str  # 用户特征文本，例如 "喜欢辣的川菜"
    top_k: int =10   

class RecommendResponse(BaseModel):
    recommended_dishes: List[str]  # 推荐的菜品名称列表

# 创建应用
app = FastAPI(title="智小厨API", description="智能菜谱推荐与聊天API服务")

# CORS配置  处理跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# # 初始化 GNNRecommender 实例
# recommender = GNNRecommender()

# 推荐 API 端点
@app.post("/recommend", response_model=RecommendResponse)


async def recommend(request: RecommendRequest):
    df = pd.read_csv("recipe.csv", encoding='utf-8')
    recipes = []
    user_features = request.user_features.split(",")  # 将用户特征字符串分割为列表
    for feature in user_features:
        if feature.endswith("菜"):
            for i in range(len(df)):
                if df.iloc[i]['地域'] == feature:
                    recipes.append(df.iloc[i]['名称'])
        else:
            for i in range(len(df)):
                if df.iloc[i]['口味'] == feature:
                    recipes.append(df.iloc[i]['名称'])
    a=np.random.randint(0, len(recipes), size=10)
    result=[]
    for i in a:
          result.append(recipes[i])

    
    top_k = request.top_k

    # 输入验证
    if not user_features or top_k <= 0:
        raise HTTPException(status_code=400, detail="用户特征不能为空且 top_k 必须大于 0")

    # 获取推荐
    try:
        # recommendations = recommender.get_recommendations(user_features, top_k=top_k)
        return RecommendResponse(recommended_dishes=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推荐过程中出错: {str(e)}")

# 启动时初始化 GNN
@app.on_event("startup")
async def startup_event():
    # recommender.initialize()
    print("GNN 模型已初始化")










# 初始化模型
llm_assistant = RAG_LLM()

# 初始化知识图谱
kg = KG.RecipeKnowledgeGraph(
    uri="neo4j://localhost:7687",
    user="neo4j",
    password="Cfy1108++.",
    database="recipedb"
)


# 请求模型
class ChatRequest(BaseModel):
    question: str
    intention:bool


# import sys
# import json
# from fastapi import FastAPI
# from starlette.responses import StreamingResponse
# import time

# @app.post("/chat1")
# async def chat1():
#     async def Stream_response():
#         # 初始消息
#         return llm_assistant.stream_response("你好")

#     return StreamingResponse(Stream_response(), media_type="text/event-stream")

# class IntentionRequest:
#     question: str

# class IntentionResponse:
#     intention: bool




# # 聊天接口 - 流式输出
# @app.post("/get_intention")
# async def get_intention(request:IntentionRequest):
#         print("意图分析")
        
#         return  IntentionResponse(intention=llm_assistant.is_cooking_intent(request.question))
        
class IntentionRequest(BaseModel):
    question: str

class IntentionResponse(BaseModel):
    intention: bool

# 聊天接口 - 流式输出
@app.post("/get_intention")
async def get_intention(request: IntentionRequest):
    print("意图分析")
    return IntentionResponse(intention=llm_assistant.is_cooking_intent(request.question))


# 聊天接口 - 流式输出
@app.post("/chat")
async def chat(request: ChatRequest):
    print(request.question)
    try:
        intention = request.intention
        if intention:
            # context1 = llm_assistant.retrieve_recipes(question=request.question, top_k=3)
            query = kg.query_recipes(request.question)
            i =query["recipes"][0]
            print(f"菜品名称：{i['name']} ,菜品得分：{i['score']} ,菜品主料：{i['main_ingredients']} , 地域：{i['regions']} ")
                
            system_prompt = kg.make_prompt(query["recipes"])
            print("系统提示词", system_prompt)
            messages = [
        {"role": "system", "content": system_prompt}, 
        { "role": "user", "content": request.question }
        ]
            # print("提示词", messages)
            response =kg.deepseek.chat.completions.create(
                model=kg.model,
                messages=messages,
                stream=True
            )
            def llm_response(a):
                for chunk in a:
                    print(chunk.choices[0].delta.content, end="")
                    yield chunk.choices[0].delta.content

            return StreamingResponse(llm_response(response), media_type="text/event-stream")
        else:
            prompt = llm_assistant.get_chat_prompt(request.question)
            messages = [
        {"role": "system", "content": prompt}, 
        { "role": "user", "content": request.question }
        ]
            # print("提示词", messages)
            response =kg.deepseek.chat.completions.create(
                model=kg.model,
                messages=messages,
                stream=True
            )
            def llm_response(a):
                for chunk in a:
                    print(chunk.choices[0].delta.content, end="")
                    yield chunk.choices[0].delta.content

            return StreamingResponse(llm_response(response), media_type="text/event-stream")
            # prompt = llm_assistant.get_chat_prompt(request.question)
        # async def Stream_response():
        #     yield f"data: {json.dumps({'intention': intention})}\n\n"
        #     yield llm_assistant.stream_response(prompt)
        #     # for chunk in stream:
        #     #     print(chunk,end="")
        #     #     yield f"data: {json.dumps({'content': chunk})}\n\n"
        #     yield f"data: {json.dumps({'event': 'done'})}\n\n"
        

            # return StreamingResponse(llm_assistant.stream_response(prompt), media_type="text/event-stream")
    except Exception as e:
        print(f"发生错误: {e}")
    #     async def error_stream():
    #         yield f"data: {json.dumps({'error': str(e)})}\n\n"
    #     return StreamingResponse(error_stream(), media_type="text/event-stream")


# @app.post("/chat")
# def chat(request: ChatRequest):
#         intention = llm_assistant.is_cooking_intent(request.question)
#         if intention:
#             context = llm_assistant.retrieve_recipes(question=request.question, top_k=3)
#             prompt = llm_assistant.get_recipe_prompt(question=request.question, context=context)
#         else:
#             prompt = llm_assistant.get_chat_prompt(request.question)


#         stream = llm_assistant.stream_response(prompt)
#         for chunk in stream:
#             yield chunk 











# 健康检查接口
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "智小厨API服务"}


# 模型信息接口
@app.get("/model-info")
def get_model_info():
    return {
        "name": "智小厨",
        "description": "一个基于大语言模型的智能烹饪助手",
        "capabilities": ["菜谱推荐", "食材搭配建议", "烹饪方法指导", "日常聊天"]
    }

class audioRequest(BaseModel):
    chunk : str






# 智能语音接口
@app.post("/Audio_agent")
def Audio_agent(request: audioRequest):
    return 







if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=1108) 



