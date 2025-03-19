import json
import numpy as np
import os
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
from langchain_community.embeddings import OllamaEmbeddings

# 1. 提取菜品特征
def extract_features(json_file_path):
    """从JSON文件中提取菜品特征"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        features = []
        names = []
        for item in data:
            feature_str = f"{item.get('taste', '')} {item.get('categories', '')}".strip()
            features.append(feature_str)
            names.append(item.get('title', 'Unknown'))
        return features, names
    except Exception as e:
        print(f"提取特征时出错: {str(e)}")
        return [], []

# 2. 生成用户特征
def generate_user_features(num_users=1000):
    """随机生成用户特征"""
    taste_options = ['甜', '咸', '酸', '辣', '麻辣', '苦', '清淡']
    cuisine_options = ['川菜', '湘菜', '粤菜', '东北菜', '沪菜', '鲁菜']
    user_features = []
    for _ in range(num_users):
        taste = np.random.choice(taste_options)
        cuisine = np.random.choice(cuisine_options)
        user_features.append(f"{taste} {cuisine}")
    return user_features

# 3. 生成交互数据
def generate_interaction_data(num_interactions=10000, num_users=1000, num_dishes=None):
    """随机生成用户-菜品交互数据"""
    interaction_data = {
        'user_id': np.random.randint(0, num_users, size=num_interactions),
        'dish_id': np.random.randint(0, num_dishes, size=num_interactions) if num_dishes else np.random.randint(0, 1000, size=num_interactions),
        'score': np.random.uniform(0, 5, size=num_interactions)
    }
    return pd.DataFrame(interaction_data)

# 4. 准备图张量
def prepare_graph_tensor(user_features, dish_features, interaction_df):
    """创建图张量，用于GNN训练"""
    num_users = user_features.shape[0]
    num_dishes = dish_features.shape[0]
    
    valid_interactions = interaction_df[
        (interaction_df['user_id'] < num_users) & 
        (interaction_df['dish_id'] < num_dishes)
    ]
    
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'user': tfgnn.NodeSet.from_fields(
                sizes=tf.constant([num_users]),
                features={'hidden_state': tf.convert_to_tensor(user_features, dtype=tf.float32)}
            ),
            'dish': tfgnn.NodeSet.from_fields(
                sizes=tf.constant([num_dishes]),
                features={'hidden_state': tf.convert_to_tensor(dish_features, dtype=tf.float32)}
            )
        },
        edge_sets={
            'interacts': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(valid_interactions)]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('user', tf.convert_to_tensor(valid_interactions['user_id'].values, dtype=tf.int32)),
                    target=('dish', tf.convert_to_tensor(valid_interactions['dish_id'].values, dtype=tf.int32))
                ),
                features={'score': tf.convert_to_tensor(valid_interactions['score'].values, dtype=tf.float32)}
            ),
            'interacts_reverse': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(valid_interactions)]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('dish', tf.convert_to_tensor(valid_interactions['dish_id'].values, dtype=tf.int32)),
                    target=('user', tf.convert_to_tensor(valid_interactions['user_id'].values, dtype=tf.int32))
                ),
                features={'score': tf.convert_to_tensor(valid_interactions['score'].values, dtype=tf.float32)}
            )
        }
    )
    return graph

# 5. 定义GNN模型
class BipartiteGNN(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim, input_dim=1024):
        super().__init__()
        # 用户节点更新
        self.user_message_fn = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.user_update_fn = tf.keras.layers.Dense(hidden_dim, activation='relu')
        # 菜品节点更新
        self.dish_message_fn = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dish_update_fn = tf.keras.layers.Dense(hidden_dim, activation='relu')
        # 初始特征投影（从输入维度到隐藏维度）
        self.user_initial_proj = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dish_initial_proj = tf.keras.layers.Dense(hidden_dim, activation='relu')
        # 最终投影
        self.user_proj = tf.keras.layers.Dense(output_dim)
        self.dish_proj = tf.keras.layers.Dense(output_dim)
    
    def call(self, graph):
        # 初始特征投影
        user_initial = self.user_initial_proj(graph.node_sets['user']['hidden_state'])
        dish_initial = self.dish_initial_proj(graph.node_sets['dish']['hidden_state'])
        
        # 用户节点更新
        user_agg = tfgnn.keras.layers.SimpleConv(
            sender_node_feature='hidden_state',
            message_fn=self.user_message_fn,
            reduce_type='sum'
        )(graph, edge_set_name='interacts_reverse')

        user_updated = self.user_update_fn(user_agg + user_initial)
        user_emb = self.user_proj(user_updated)
        
        # 菜品节点更新
        dish_agg = tfgnn.keras.layers.SimpleConv(
            sender_node_feature='hidden_state',
            message_fn=self.dish_message_fn,
            reduce_type='sum'
        )(graph, edge_set_name='interacts')

        dish_updated = self.dish_update_fn(dish_agg + dish_initial)
        dish_emb = self.dish_proj(dish_updated)
        
        return user_emb, dish_emb

# 6. 训练相关函数
def compute_loss(model, graph):
    """计算损失函数"""
    user_emb, dish_emb = model(graph)
    edge_indices = graph.edge_sets['interacts'].adjacency
    user_idx = edge_indices.source
    dish_idx = edge_indices.target
    pred_scores = tf.reduce_sum(
        tf.gather(user_emb, user_idx) * tf.gather(dish_emb, dish_idx),
        axis=1
    )
    true_scores = graph.edge_sets['interacts']['score']
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(true_scores, pred_scores))

def train_model(model, graph, epochs=100, lr=0.01):
    """训练模型"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, graph)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f'第 {epoch} 轮, 损失: {loss.numpy():.4f}')
    return model

def get_recommendations(model, graph, user_id, top_k=5):
    """获取推荐结果"""
    user_emb, dish_emb = model(graph)
    user_vec = tf.nn.l2_normalize(user_emb[user_id], axis=0)
    scores = tf.matmul(dish_emb, tf.expand_dims(user_vec, 1))[:, 0]
    top_indices = tf.argsort(scores, direction='DESCENDING')[:top_k]
    return top_indices.numpy()

# 7. 保存和加载函数
def save_features(user_features, dish_features, names, save_dir="saved_data"):
    """保存用户特征和菜品特征"""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "user_features.npy"), user_features)
    np.save(os.path.join(save_dir, "dish_features.npy"), dish_features)
    with open(os.path.join(save_dir, "dish_names.pkl"), "wb") as f:
        pickle.dump(names, f)
    print(f"特征和名称已保存到 {save_dir}")

def load_features(save_dir="saved_data"):
    """加载用户特征和菜品特征"""
    user_features = np.load(os.path.join(save_dir, "user_features.npy"))
    dish_features = np.load(os.path.join(save_dir, "dish_features.npy"))
    with open(os.path.join(save_dir, "dish_names.pkl"), "rb") as f:
        names = pickle.load(f)
    print(f"已从 {save_dir} 加载特征和名称")
    return user_features, dish_features, names

def save_interaction_data(interaction_df, save_dir="saved_data"):
    """保存交互数据"""
    os.makedirs(save_dir, exist_ok=True)
    interaction_df.to_csv(os.path.join(save_dir, "interactions.csv"), index=False)
    print(f"交互数据已保存到 {os.path.join(save_dir, 'interactions.csv')}")

def load_interaction_data(save_dir="saved_data"):
    """加载交互数据"""
    interaction_df = pd.read_csv(os.path.join(save_dir, "interactions.csv"))
    print(f"已加载交互数据，形状: {interaction_df.shape}")
    return interaction_df

def save_model(model, save_dir="saved_model"):
    """保存训练好的模型"""
    os.makedirs(save_dir, exist_ok=True)
    model.save_weights(os.path.join(save_dir, "model_weights"))
    print(f"模型已保存到 {os.path.join(save_dir, 'model_weights')}")

def load_model(hidden_dim=512, output_dim=128, save_dir="saved_model"):
    """加载训练好的模型"""
    model = BipartiteGNN(hidden_dim, output_dim)
    dummy_graph = prepare_dummy_graph(1024)  # 匹配初始特征维度
    _ = model(dummy_graph)  # 初始化模型权重
    model.load_weights(os.path.join(save_dir, "model_weights"))
    print(f"已加载模型权重")
    return model

def prepare_dummy_graph(dim):
    """创建虚拟图用于初始化模型"""
    dummy_user_features = np.random.random((2, dim)).astype(np.float32)
    dummy_dish_features = np.random.random((2, dim)).astype(np.float32)
    dummy_interactions = pd.DataFrame({
        'user_id': [0, 1],
        'dish_id': [0, 1],
        'score': [0.5, 0.5]
    })
    return prepare_graph_tensor(dummy_user_features, dummy_dish_features, dummy_interactions)

def save_updated_features(model, graph, save_dir="saved_data"):
    """保存更新后的特征"""
    os.makedirs(save_dir, exist_ok=True)
    user_emb, dish_emb = model(graph)
    np.save(os.path.join(save_dir, "updated_user_features.npy"), user_emb.numpy())
    np.save(os.path.join(save_dir, "updated_dish_features.npy"), dish_emb.numpy())
    print(f"更新后的特征已保存到 {save_dir}")


def load_user_features_text(save_dir="saved_data"):
    """加载用户原始文本特征"""
    text_file = os.path.join(save_dir, "user_features_text.pkl")
    if os.path.exists(text_file):
        with open(text_file, "rb") as f:
            user_features_text = pickle.load(f)
        print(f"已从 {text_file} 加载用户特征文本")
        return user_features_text
    else:
        print(f"未找到 {text_file}，将生成新特征")
        return None





# 8. 主函数
def main():
    data_save_dir = "saved_data"
    model_save_dir = "saved_model"
    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    embeddings = OllamaEmbeddings(model="smartcreation/bge-large-zh-v1.5:latest")
    
    # 加载或生成数据
    if os.path.exists(os.path.join(data_save_dir, "user_features.npy")):
        print("特征数据已存在，直接加载")
        user_features, dish_features, names = load_features(data_save_dir)
        interaction_df = load_interaction_data(data_save_dir)
    else:
        print("特征数据不存在，正在生成")
        print("正在生成菜品特征")
        json_file = "all_recipe.json"
        dish_features_text, names = extract_features(json_file)
        dish_features = np.array([embeddings.embed_query(text) for text in dish_features_text])
        print(f'菜品数据已生成{dish_features.shape}')
        
        print("正在生成用户特征")
        user_features_text = generate_user_features()  # 保留原始文本特征
        user_features = np.array([embeddings.embed_query(text) for text in user_features_text])
        print(f'用户数据已生成{user_features.shape}')
        
        interaction_df = generate_interaction_data(num_dishes=len(dish_features))
        
        save_features(user_features, dish_features, names, data_save_dir)
        save_interaction_data(interaction_df, data_save_dir)
        # 保存原始文本特征
        with open(os.path.join(data_save_dir, "user_features_text.pkl"), "wb") as f:
            pickle.dump(user_features_text, f)
        print(f"用户原始文本特征已保存到 {os.path.join(data_save_dir, 'user_features_text.pkl')}")
    
    # 准备图数据
    graph_tensor = prepare_graph_tensor(user_features, dish_features, interaction_df)
    print(f"GraphTensor: {graph_tensor}")
    print(f'用户特征：{user_features.shape}')
    print(f'菜品特征：{dish_features.shape}')
    
    # 加载或训练模型
    model = BipartiteGNN(hidden_dim=512, output_dim=128)
    if os.path.exists(os.path.join(model_save_dir, "model_weights.index")):
        model.load_weights(os.path.join(model_save_dir, "model_weights"))
        print("已加载保存的模型")
    else:
        print("模型正在训练")
        model = train_model(model, graph_tensor)
        save_model(model, model_save_dir)
    
    # 保存更新后的特征
    save_updated_features(model, graph_tensor, data_save_dir)
    
# 获取推荐并输出用户 0 的相关信息
    user_id = 0
    recommendations = get_recommendations(model, graph_tensor, user_id)
    recommended_dishes = [names[i] for i in recommendations]
    print(f"\n用户 {user_id} 的推荐菜品: {recommended_dishes}")

    # 获取用户 0 的特征文本
    user_features_text = load_user_features_text()
    print(f'用户 {user_id} 的特征: {user_features_text[0]}')

    # 获取所有菜品的特征文本
    dish_features_text, dish_names = extract_features('all_recipe.json')
    print(f'推荐菜品的特征: {[dish_features_text[i] for i in recommendations]}')

    # 获取用户 0 的交互数据
    inter_0 = interaction_df[interaction_df['user_id'] == 0]
    inter_dish = inter_0['dish_id']
    inter_score = inter_0['score']

    # 输出用户 0 交互过的菜品的特征和得分
    print(f"\n用户 {user_id} 的交互数据（共 {len(inter_0)} 条）：")
    for dish_id, score in zip(inter_dish, inter_score):
        dish_feature = dish_features_text[dish_id] if dish_id < len(dish_features_text) else "未知特征"
        dish_name = dish_names[dish_id] if dish_id < len(dish_names) else "未知菜品"
        print(f"菜品 ID: {dish_id}, 名称: {dish_name}, 特征: {dish_feature}, 得分: {score:.4f}")
    

if __name__ == "__main__":
    main()