import json
import re
from dotenv import load_dotenv
load_dotenv()
# Markdown 格式的输入（包含 JSON 代码块）
import os 

from openai import OpenAI




def extract_json_from_markdown(markdown_text):
    # 匹配```json ... ```代码块
    match = re.search(r'```json\s*(.*?)\s*```', markdown_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json.loads(json_str)
    else:
        raise ValueError("未找到JSON代码块")


openai_api_key = os.getenv("OPENAI_API_KEY") # 读取 OpenAI API Key
base_url = os.getenv("BASE_URL") # 读取 BASE YRL
model = os.getenv("MODEL") # 读取 model

deepseek=OpenAI(api_key=openai_api_key, base_url=base_url)

user_query = "我想吃宫保鸡丁，喜欢辣的川菜，适合夏天。"
messages = [
        {
            "role": "system",
            "content": """你是一个智能助手，擅长从用户查询中提取与菜谱相关的实体。你的任务是从用户输入中识别以下七类实体，并以有效的 JSON 格式返回结果。输出的 JSON 必须包含以下字段：recipe_name（列表）、ingredients（列表）、tastes（列表）、regions（列表）、methods（列表）、seasons（列表）、热量（对象）。

### 实体定义和规则
1. **菜名（recipe_name）**：菜的名称，提取用户明确提到的菜名。如果没有提到，返回空列表 `[]`。
2. **菜的用料（ingredients）**：菜的主要食材，提取用户提到的食材。如果没有提到，返回空列表 `[]`。
3. **口味（tastes）**：菜的味道，必须从以下参考列表中选择：
   - 参考列表：["咸甜", "微辣", "甜味", "咖喱", "咸鲜", "酸甜", "清淡", "酸咸", "酸辣", "蒜香", "其他", "原味", "中辣", "葱香", "奶香", "超辣", "酱香", "麻辣", "五香", "果味", "甜香", "香辣", "咸香", "鱼香", "孜然", "苦味", "香草", "糟香", "怪味", "麻香"]
   - 如果用户输入的口味明确匹配列表中的值，直接使用。
   - 如果输入模糊（例如“甜的”），选择语义最接近的参考值（如“甜味”）。
   - 如果无法匹配，返回空列表 `[]`。
4. **地域（regions）**：菜的来源，必须从以下参考列表中选择：
   - 参考列表：["苏菜", "川菜", "浙菜", "湘菜", "粤菜", "西北菜", "新疆菜", "东北菜", "陕西菜", "鲁菜", "淮扬菜", "潮汕菜", "北京菜", "鄂菜", "上海菜", "河北菜", "重庆菜", "豫菜", "晋菜", "赣菜", "闽菜", "藏菜", "徽菜", "贵州菜", "云南菜"]
   - 如果用户输入明确匹配列表中的值，直接使用。
   - 如果输入模糊（例如“四川菜”），映射到对应的值（如“川菜”）。
   - 如果无法匹配，返回空列表 `[]`。
5. **方法（methods）**：菜的烹饪方法，必须从以下参考列表中选择：
   - 参考列表：["炖", "扒", "腌", "烧", "焖", "炸", "火锅", "煎", "烤", "熏", "冷冻", "调味", "生鲜", "卤", "炒", "氽", "煨", "煮", "蒸", "拔丝", "拌", "煲", "酥", "砂锅", "溜", "爆", "烙", "烘焙", "酱"]
   - 如果用户输入明确提到烹饪方法，直接使用。
   - 如果输入模糊（例如“烧的”），选择语义最接近的参考值（如“烧”或“炖”）。
   - 如果无法匹配，返回空列表 `[]`。
6. **季节（seasons）**：菜的适合季节或节日，必须从以下参考列表中选择：
   - 参考列表：["元宵节", "秋季食谱", "儿童节", "冬季食谱", "夏季食谱", "立夏", "春季食谱", "立春", "中秋", "端午节", "圣诞节", "万圣节", "情人节", "七夕", "惊蛰", "年夜饭", "冬至", "寒露", "芒种", "处暑", "复活节", "小满", "二月二", "大雪", "小暑", "立秋", "雨水", "夏至", "清明", "秋分", "立冬", "霜降", "重阳节", "感恩节", "小寒", "大寒"]
   - 如果用户输入明确提到季节或节日，直接使用。
   - 如果输入模糊（例如“热天”），映射到对应的值（如“夏季食谱”）。
   - 如果无法匹配，返回空列表 `[]`。
7. **热量（热量）**：菜的热量范围，格式为 JSON 对象：
   - 如果用户提到热量范围（例如“低于400卡”），返回 `{"operator": "<", "value": 400}`。
   - 支持的 operator："<", ">", "=", "≤", "≥"。
   - 如果没有提到热量，返回 `{}`。

### 输出要求
- 输出必须是有效的 JSON 格式，字段顺序固定为：recipe_name, ingredients, tastes, regions, methods, seasons, 热量。
- 所有列表字段（recipe_name, ingredients, tastes, regions, methods, seasons）的值必须是字符串列表。
- 热量字段必须是 JSON 对象，格式为 {"operator": "<", "value": 400} 或 {}。

### 示例输入输出
**输入**：我想吃宫保鸡丁，喜欢辣的川菜，适合夏天。  
**输出**:
```
{
    "recipe_name": ["宫保鸡丁"],
    "ingredients": ["鸡肉"],
    "tastes": ["麻辣"],
    "regions": ["川菜"],
    "methods": [],
    "seasons": ["夏季食谱"],
    "热量": {}
}
```

""" }, 
    { "role": "user", "content": user_query } ]     





response = deepseek.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
)
llm_response=response
for chunk in llm_response:
    print(chunk.choices[0].delta.content, end="")


# a=extract_json_from_markdown(llm_response)
# print(a)












# messages = [
#         {"role": "system", "content": 1 }, 
#         { "role": "user", "content": user_query }
#     ]