import pandas as pd
from neo4j import GraphDatabase
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
# 加载 .env 文件，确保 API Key 受到保护
load_dotenv()
import json


# 1. 连接到 Neo4j 数据库
class RecipeKnowledgeGraph:
    def __init__(self, uri, user, password, database="recipedb"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
        # deepseek 实现NER
        self.openai_api_key = os.getenv("OPENAI_API_KEY") # 读取 OpenAI API Key
        self.base_url = os.getenv("BASE_URL") # 读取 BASE YRL
        self.model = os.getenv("MODEL") # 读取 model
        if not self.openai_api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置OPENAI_API_KEY")
        else :
            self.deepseek=OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
            print("Deepseek已就位")

        # 定义常见实体字典
        self.recipe_name=pd.read_csv("recipe.csv",encoding='utf-8-sig')['名称'].tolist()
        self.tastes = pd.read_csv('tastes.csv', encoding='utf-8-sig')['口味'].tolist()
        self.regions = pd.read_csv('region.csv', encoding='utf-8-sig')['地域'].tolist()
        self.methods = pd.read_csv('methods.csv', encoding='utf-8-sig')['烹饪方法'].tolist()
        self.seasons = pd.read_csv('seasons.csv', encoding='utf-8-sig')['季节'].tolist()
        
        
        # if '辣椒炒肉' in self.recipe_name:
        #     print(1)
        # if '鱼香肉丝' in self.recipe_name:
        #     print(2)

        # print(self.recipe_name[0:10])
        # print(self.tastes[0:10])
        # print(self.regions[0:10])
        print("已加载常见实体字典")



    def close(self):
        self.driver.close()

    # 2. 清空数据库
    def clear_database(self):
        query = "MATCH (n) DETACH DELETE n"
        with self.driver.session(database=self.database) as session:
            session.run(query)
        print("数据库已清空")

    # 3. 导入 CSV 数据并构建知识图谱   
    def import_recipes_from_csv(self, csv_file):
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        print("CSV 文件的列名：", df.columns.tolist())

        with self.driver.session(database=self.database) as session:
            for index, row in df.iterrows():
                recipe_id = index+1  
                main_ingredients = str(row['主料']) if not pd.isna(row['主料']) else ''
                secondary_ingredients = str(row['辅料']) if not pd.isna(row['辅料']) else ''
                seasonings = str(row['调料']) if not pd.isna(row['调料']) else ''
                taste = str(row['口味']) if not pd.isna(row['口味']) else ''
                method = str(row['方法']) if not pd.isna(row['方法']) else ''
                region = str(row['地域']) if not pd.isna(row['地域']) else ''
                season = str(row['季节']) if not pd.isna(row['季节']) else ''
                steps = str(row['步骤']) if not pd.isna(row['步骤']) else ''
                calories = str(row['热量(卡路里)']) if not pd.isna(row['热量(卡路里)']) else ''
                nutrition = str(row['营养']) if not pd.isna(row['营养']) else ''

                # Combine secondary ingredients and seasonings
                combined_seasonings = (secondary_ingredients + ' ' + seasonings).strip()
                

                recipe_query = """
                MERGE (r:Recipe {name: $name})
                SET r.RecipeID = $recipe_id,
                    r.main_ingredients = $main_ingredients, 
                    r.seasonings = $combined_seasonings, 
                    r.taste = $taste, 
                    r.method = $method, 
                    r.region = $region, 
                    r.season = $season, 
                    r.steps = $steps, 
                    r.calories = $calories, 
                    r.nutrition = $nutrition
                """
                session.run(recipe_query, 
                            name=row['名称'], 
                            recipe_id=recipe_id,
                            main_ingredients=main_ingredients,
                            combined_seasonings=combined_seasonings,
                            taste=taste,
                            method=method,
                            region=region,
                            season=season,
                            steps=steps,
                            calories=calories,
                            nutrition=nutrition)

                if main_ingredients:
                    for ingredient in main_ingredients.split():
                        ingredient = ingredient.strip()
                        if ingredient:
                            ingredient_query = """
                            MERGE (i:Ingredient {name: $name})
                            WITH i
                            MATCH (r:Recipe {name: $recipe_name})
                            MERGE (r)-[:HAS_MAIN_INGREDIENT]->(i)
                            """
                            session.run(ingredient_query, name=ingredient, recipe_name=row['名称'])

                if combined_seasonings:
                    for seasoning in combined_seasonings.split():
                        seasoning = seasoning.strip()
                        if seasoning:
                            seasoning_query = """
                            MERGE (s:Seasoning {name: $name})
                            WITH s
                            MATCH (r:Recipe {name: $recipe_name})
                            MERGE (r)-[:USES_SEASONING]->(s)
                            """
                            session.run(seasoning_query, name=seasoning, recipe_name=row['名称'])

                if taste:
                    for single_taste in taste.split():
                        single_taste = single_taste.strip()
                        if single_taste:
                            # 处理口味，避免重复添加
                            if single_taste not in self.tastes:
                                self.tastes.append(single_taste)

                            taste_query = """
                                MERGE (t:Taste {name: $name})
                                WITH t
                                MATCH (r:Recipe {name: $recipe_name})
                                MERGE (r)-[:HAS_TASTE]->(t)
                                """
                            session.run(taste_query, name=single_taste, recipe_name=row['名称'])

                if method:
                    for single_method in method.split():
                        single_method = single_method.strip()
                        if single_method:
                            if single_method not in self.methods:
                                self.methods.append(single_method)
                            method_query = """
                                MERGE (m:Method {name: $name})
                                WITH m
                                MATCH (r:Recipe {name: $recipe_name})
                                MERGE (r)-[:USES_METHOD]->(m)
                                """
                            session.run(method_query, name=single_method, recipe_name=row['名称'])

                if region:
                    for single_region in region.split():
                        single_region = single_region.strip()
                        if single_region:
                            if single_region not in self.regions:
                                self.regions.append(single_region)
                           
                            region_query = """
                            MERGE (reg:Region {name: $name})
                            WITH reg
                            MATCH (r:Recipe {name: $recipe_name})
                            MERGE (r)-[:FROM_REGION]->(reg)
                            """
                            session.run(region_query, name=single_region, recipe_name=row['名称'])

                if season:
                    for single_season in season.split():
                        single_season = single_season.strip()
                        if single_season:
                            if single_season not in self.seasons:
                                self.seasons.append(single_season)

                            season_query = """
                            MERGE (s:Season {name: $name})
                            WITH s
                            MATCH (r:Recipe {name: $recipe_name})
                            MERGE (r)-[:SUITABLE_FOR_SEASON]->(s)
                            """
                            session.run(season_query, name=single_season, recipe_name=row['名称'])

        print("知识图谱构建完成，所有菜品已生成编号（使用 index 作为 RecipeID）")
        # df = pd.DataFrame(self.tastes, columns=['口味'])
        # df.to_csv('tastes.csv', index=False, encoding='utf-8')

        # df = pd.DataFrame(self.seasons, columns=['季节'])
        # df.to_csv('seasons.csv', index=False, encoding='utf-8')
        
        # df = pd.DataFrame(self.methods, columns=['烹饪方法'])
        # df.to_csv('methods.csv', index=False, encoding='utf-8')

        # df = pd.DataFrame(self.regions, columns=['地域'])
        # df.to_csv('region.csv', index=False, encoding='utf-8')

    
    def extract_json_from_markdown(self,markdown_text):
        # 匹配```json ... ```代码块
        match = re.search(r'```json\s*(.*?)\s*```', markdown_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            raise ValueError("未找到JSON代码块")
    
    def ner_prompt(self, user_query):
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

        # print(messages)
        return messages
    
    def ner_response(self, user_query):
        messages = self.ner_prompt(user_query)
        print(f'当前进行:NER ing:')
        response = self.deepseek.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        llm_response = response.choices[0].message.content
        entities = self.extract_json_from_markdown(llm_response)
        print(f'ner实体提取结果:')
        print(entities)
        # 二次筛选
        entities["tastes"] = [t for t in entities.get("tastes", []) if t in self.tastes]
        entities["regions"] = [r for r in entities.get("regions", []) if r in self.regions]
        entities["methods"] = [m for m in entities.get("methods", []) if m in self.methods]
        entities["seasons"] = [s for s in entities.get("seasons", []) if s in self.seasons]
        return entities

    


    # 5. 知识图谱检索

    def query_recipes(self, user_query):
        try:
            # 提取实体
            entities = self.ner_response(user_query)
            print('开始进行知识图谱查询')

            # 限制每类实体最多取 3 个
            def top3(lst):
                return lst[:3] if lst else []

            recipe_names = top3(entities.get("recipe_name", []))
            ingredients = top3(entities.get("ingredients", []))
            tastes = top3(entities.get("tastes", []))
            regions = top3(entities.get("regions", []))
            methods = top3(entities.get("methods", []))
            seasons = top3(entities.get("seasons", []))
            calories = entities.get("热量", {})

            # 构建参数字典
            params = {
                "recipe_names": recipe_names,
                "ingredients": ingredients,
                "tastes": tastes,
                "regions": regions,
                "methods": methods,
                "seasons": seasons,
                "calories_operator": calories.get("operator"),
                "calories_value": calories.get("value")
            }

            # 确保至少有一个非空实体
            if not any([recipe_names, ingredients, tastes, regions, methods, seasons, calories.get("operator")]):
                return {"error": "查询过于模糊，无法匹配菜谱", "recipes": []}

            # 构建 Cypher 查询
            query = """
            MATCH (r:Recipe)
            OPTIONAL MATCH (r)-[:HAS_MAIN_INGREDIENT]->(mi:Ingredient)
            OPTIONAL MATCH (r)-[:USES_SEASONING]->(s:Seasoning)
            OPTIONAL MATCH (r)-[:HAS_TASTE]->(t:Taste)
            OPTIONAL MATCH (r)-[:USES_METHOD]->(m:Method)
            OPTIONAL MATCH (r)-[:FROM_REGION]->(reg:Region)
            OPTIONAL MATCH (r)-[:SUITABLE_FOR_SEASON]->(sea:Season)

            WITH r,
                 collect(DISTINCT mi.name) AS main_ingredients,
                 collect(DISTINCT s.name) AS seasonings,
                 collect(DISTINCT t.name) AS tastes,
                 collect(DISTINCT m.name) AS methods,
                 collect(DISTINCT reg.name) AS regions,
                 collect(DISTINCT sea.name) AS seasons

            WHERE ($calories_operator IS NULL OR
                  ($calories_operator = "<" AND toInteger(r.calories) < $calories_value) OR
                  ($calories_operator = ">" AND toInteger(r.calories) > $calories_value) OR
                  ($calories_operator = "=" AND toInteger(r.calories) = $calories_value) OR
                  ($calories_operator = "≤" AND toInteger(r.calories) <= $calories_value) OR
                  ($calories_operator = "≥" AND toInteger(r.calories) >= $calories_value))

            WITH r, main_ingredients, seasonings, tastes, methods, regions, seasons,

                // 权重打分逻辑
                REDUCE(score = 0.0,
                    name IN $recipe_names |
                    score + CASE
                        WHEN r.name = name THEN 5.0
                        WHEN toLower(r.name) CONTAINS toLower(name) THEN 3.0
                        ELSE 0.0 END
                ) +
                REDUCE(score = 0.0,
                    ing IN $ingredients |
                    score + CASE
                        WHEN ANY(mi IN main_ingredients WHERE toLower(mi) CONTAINS toLower(ing)) THEN 2.0
                        WHEN ANY(s IN seasonings WHERE toLower(s) CONTAINS toLower(ing)) THEN 1.0
                        ELSE 0.0 END
                ) +
                REDUCE(score = 0.0,
                    taste IN $tastes | score + CASE WHEN taste IN tastes THEN 2.0 ELSE 0.0 END
                ) +
                REDUCE(score = 0.0,
                    reg IN $regions | score + CASE WHEN reg IN regions THEN 1.5 ELSE 0.0 END
                ) +
                REDUCE(score = 0.0,
                    mtd IN $methods | score + CASE WHEN mtd IN methods THEN 1.5 ELSE 0.0 END
                ) +
                REDUCE(score = 0.0,
                    sea IN $seasons | score + CASE WHEN sea IN seasons THEN 1.5 ELSE 0.0 END
                ) +
                CASE WHEN $calories_operator IS NOT NULL AND 
                          ((($calories_operator = "<" AND toInteger(r.calories) < $calories_value) OR
                            ($calories_operator = ">" AND toInteger(r.calories) > $calories_value) OR
                            ($calories_operator = "=" AND toInteger(r.calories) = $calories_value) OR
                            ($calories_operator = "≤" AND toInteger(r.calories) <= $calories_value) OR
                            ($calories_operator = "≥" AND toInteger(r.calories) >= $calories_value)))
                          THEN 1.0 ELSE 0.0 END
                AS score

            WHERE score > 0
            ORDER BY score DESC
            LIMIT 5

            RETURN r,
                   main_ingredients,
                   seasonings,
                   tastes,
                   methods,
                   regions,
                   seasons,
                   score
            """

            with self.driver.session(database=self.database) as session:
                results = session.run(query, **params)
                recipes = []
                for record in results:
                    recipe_node = record["r"]
                    recipe = {
                        "recipe_id": recipe_node.get("RecipeID"),
                        "name": recipe_node.get("name"),
                        "main_ingredients": record["main_ingredients"],
                        "seasonings": record["seasonings"],
                        "tastes": record["tastes"],
                        "methods": record["methods"],
                        "regions": record["regions"],
                        "seasons": record["seasons"],
                        "steps": recipe_node.get("steps"),
                        "calories": recipe_node.get("calories"),
                        "nutrition": recipe_node.get("nutrition"),
                        "score": record["score"]
                    }
                    recipes.append(recipe)

                if not recipes:
                    return {"error": "未找到匹配的菜谱", "recipes": []}
                return {"error": None, "recipes": recipes}

        except Exception as e:
            print(f"查询失败：{e}")
            return {"error": f"查询失败：{str(e)}", "recipes": []}

    def make_prompt(self, query):
        return f"""你是一位专业的烹饪助手“智小厨”，请基于以下知识库信息回答用户问题。

    ### 知识库信息:
    {query}

    ### 任务要求:
    1. 仔细分析用户需求和提供的菜谱信息，判断用户是否需要推荐菜谱。
       - 如果用户需要推荐菜谱，并且知识库中有合适的菜谱，请选择最合适的1-3个菜谱进行推荐。
       - 如果用户需要推荐菜谱，但知识库中没有合适的菜谱，菜品id为空，直接用你的烹饪知识进行回复。
       - 如果用户的问题不需要推荐菜谱（如只问烹饪技巧、食材处理、营养建议等），菜品id为空，直接用你的烹饪知识进行回复。
    2. 返回结果必须是一个 JSON 格式的字典，包含以下两个字段：
       - "dishes_id": 一个列表，包含所有推荐的菜品id（1-3个），每个菜名为简洁的字符串。如果不需要推荐菜谱或没有合适菜谱，则该列表为空。
       - "response": 一个字符串，面向用户的完整回复，使用 Markdown 格式，既可以是推荐理由，也可以是对用户问题的专业解答。
    3. 输出必须是有效的 JSON 字符串。
    4. 推荐理由或回复要自然流畅，无需提及知识库内容或数据来源。

    ### 示例:
    - 用户问题: "推荐一个川菜"
      输出: {{"dishes_id": [118], "response": "### 推荐理由\n麻婆豆腐是经典川菜，麻辣鲜香，适合喜欢辣味的用户！"}}
    - 用户问题: "推荐两个清淡的菜"
      输出: {{"dishes_id": [516, 5], "response": "### 推荐理由\n清蒸鲈鱼口感鲜嫩，保留食材原味；冬瓜猪肉汤清热解暑，简单易做，都非常清淡健康！"}}
    - 用户问题: "如何保存新鲜的鱼？"
      输出: {{"dishes_id": [], "response": "### 智小厨小贴士\n新鲜的鱼应尽快冷藏或冷冻，冷藏时可用保鲜膜包裹并放入冰箱冷藏室，冷冻则需去除内脏后密封保存，以保持鱼肉的新鲜和口感。"}}
    - 用户问题: "推荐一个低脂的粤菜"
      输出: {{"dishes_id": [], "response": "### 推荐建议\n很抱歉，当前知识库中没有现成的菜谱卡片推荐。但是我知道有一个- 白切鸡（去皮食用）特点：粤菜经典，鸡肉煮熟后冰镇，皮爽肉嫩，脂肪主要集中在皮上，去皮后低脂高蛋白。- 低脂关键:关键选择走地鸡（脂肪较少），食用时去掉鸡皮，蘸姜葱酱或酱油而非油腻酱料。"}}

    注意示例中的dishes_id具体值来源于知识库
    请严格按照上述格式和要求返回 JSON 字符串。
    """
        
        







def main():
    kg = RecipeKnowledgeGraph(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="Cfy1108++.",
        database="recipedb"
    )

    user_query = "我想吃猪肉，喜欢辣的川菜，请给我推荐一个菜。"
    print(f"用户查询：{user_query}")
    a=kg.ner_response(user_query)
    print(f'NER结果：{a}')

    # result = kg.query_recipes(user_query)

    # if result["error"]:
    #     print(result["error"])
    # else:
    #     # for i in result["recipes"]:
    #     #     print(f"菜品名称：{i['name']} ,菜品得分：{i['score']} ,菜品主料：{i['main_ingredients']} , 地域：{i['regions']} ")
    #     # # prompt = kg.build_prompt(user_query, result["recipes"])
    #     a=kg.make_prompt(user_query)
    #     print(a)
    kg.close()

if __name__ == "__main__":
    main()