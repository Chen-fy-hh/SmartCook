import json
import numpy as np
import os
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
from langchain_community.embeddings import OllamaEmbeddings

class GNNRecommender:
    def __init__(self):
        self.model = None
        self.graph_tensor = None
        self.names = None
        self.embeddings = OllamaEmbeddings(model="smartcreation/bge-large-zh-v1.5:latest")
        self.data_save_dir = "saved_data"
        self.model_save_dir = "saved_model"
        df = pd.read_csv('D:\\NLP\\DPR1\\recipe.csv', encoding='utf-8')
        self.taste=df['口味'].unique().tolist()
        self.catagories=df['地域'].unique().tolist()
        print(self.taste)
        print(self.catagories)

    def extract_features(self, csv_file_path):
        """从 CSV 文件中提取菜品特征"""
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8')

            features = []
            names = []
            for _, row in df.iterrows():
                taste = str(row.get('口味', ''))  
                # print(taste)
                categories = str(row.get('地域', '')) 
                # print(categories)
                feature_str = f"{taste} {categories}".strip()  
                features.append(feature_str)
                name = str(row.get('名称', 'Unknown'))
                # print(name)
                names.append(name)
            print(f"成功从 {csv_file_path} 提取 {len(features)} 条菜品特征")
            return features, names
        except Exception as e:
            print(f"提取特征时出错: {str(e)}")
            return [], []

    def generate_user_features(self, num_users=5000):
        """生成随机用户特征"""

        taste_options = self.taste
        cuisine_options = self.catagories

        user_features = [f"{np.random.choice(taste_options)} {np.random.choice(cuisine_options)}" 
                         for _ in range(num_users)]
        print(f"生成 {num_users} 个用户特征")
        return user_features

    def generate_interaction_data(self, num_interactions=10000, num_users=1000, num_dishes=1000):
        """生成随机交互数据"""
        interaction_data = {
            'user_id': np.random.randint(0, num_users, size=num_interactions),
            'dish_id': np.random.randint(0, num_dishes, size=num_interactions),
            'score': np.random.uniform(0, 5, size=num_interactions)
        }
        print(f"生成 {num_interactions} 条交互数据")
        return pd.DataFrame(interaction_data)

    def prepare_graph_tensor(self, user_features, dish_features, interaction_df):
        """创建图张量"""
        num_users = len(user_features)
        num_dishes = len(dish_features)
        
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
        print(f"图张量创建完成：{num_users} 个用户，{num_dishes} 个菜品，{len(valid_interactions)} 条交互边")
        return graph

    def initialize(self):
        """初始化模型和数据"""
        os.makedirs(self.data_save_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

        # 强制重新生成数据（模拟没有保存的数据）
        print("开始生成新数据...")
        user_features_text = self.generate_user_features(5000)
        dish_features_text, self.names = self.extract_features("recipe.csv")
        if not dish_features_text:
            print("未找到 recipe.csv，生成随机菜品特征")
            dish_features_text = self.generate_user_features(1000)
            self.names = [f"菜品{i}" for i in range(1000)]
        
        print("生成嵌入向量...")
        user_features = np.array(self.embeddings.embed_documents(user_features_text))
        dish_features = np.array(self.embeddings.embed_documents(dish_features_text))
        interaction_df = self.generate_interaction_data(10000, 5000, len(self.names))
        
        self.save_features(user_features, dish_features, self.names)
        self.save_interaction_data(interaction_df)
        
        self.graph_tensor = self.prepare_graph_tensor(user_features, dish_features, interaction_df)

        print("初始化模型...")
        self.model = BipartiteGNN(hidden_dim=512, output_dim=128, input_dim=user_features.shape[1])
        print("开始训练模型...")
        self.model = self.train_model(self.model, self.graph_tensor)
        self.save_model(self.model)
        print("模型训练完成并保存")

    def get_recommendations(self, user_features_text, top_k=5):
        """获取推荐菜品"""
        user_emb_text = self.embeddings.embed_query(user_features_text)
        user_emb_text = np.array([user_emb_text], dtype=np.float32)
        
        user_emb, dish_emb = self.model(self.graph_tensor)
        
        if user_emb_text.shape[1] != dish_emb.shape[1]:
            projection_layer = tf.keras.layers.Dense(dish_emb.shape[1])
            user_emb_text = projection_layer(user_emb_text)
        
        scores = tf.matmul(dish_emb, tf.transpose(user_emb_text))
        top_indices = tf.argsort(scores[:, 0], direction='DESCENDING')[:top_k].numpy()
        
        return [self.names[idx] for idx in top_indices]

    def save_features(self, user_features, dish_features, names):
        np.save(os.path.join(self.data_save_dir, "user_features.npy"), user_features)
        np.save(os.path.join(self.data_save_dir, "dish_features.npy"), dish_features)
        with open(os.path.join(self.data_save_dir, "dish_names.pkl"), "wb") as f:
            pickle.dump(names, f)
        print("特征数据已保存")

    def load_features(self):
        user_features = np.load(os.path.join(self.data_save_dir, "user_features.npy"))
        dish_features = np.load(os.path.join(self.data_save_dir, "dish_features.npy"))
        with open(os.path.join(self.data_save_dir, "dish_names.pkl"), "rb") as f:
            names = pickle.load(f)
        return user_features, dish_features, names

    def save_interaction_data(self, interaction_df):
        interaction_df.to_csv(os.path.join(self.data_save_dir, "interactions.csv"), index=False)
        print("交互数据已保存")

    def load_interaction_data(self):
        return pd.read_csv(os.path.join(self.data_save_dir, "interactions.csv"))

    def save_model(self, model):
        model.save_weights(os.path.join(self.model_save_dir, "model_weights"))
        print("模型权重已保存")

    def train_model(self, model, graph, epochs=200, lr=0.01):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.compute_loss(model, graph)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(f'第 {epoch} 轮, 损失: {loss.numpy():.4f}')
        return model

    def compute_loss(self, model, graph):
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

class BipartiteGNN(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim, input_dim=1024):
        super().__init__()
        self.user_initial_proj = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dish_initial_proj = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.user_message_fn = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.user_update_fn = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dish_message_fn = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dish_update_fn = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.user_proj = tf.keras.layers.Dense(output_dim)
        self.dish_proj = tf.keras.layers.Dense(output_dim)
    
    def call(self, graph):
        user_initial = self.user_initial_proj(graph.node_sets['user']['hidden_state'])
        dish_initial = self.dish_initial_proj(graph.node_sets['dish']['hidden_state'])
        
        user_agg = tfgnn.keras.layers.SimpleConv(
            sender_node_feature='hidden_state',
            message_fn=self.user_message_fn,
            reduce_type='sum'
        )(graph, edge_set_name='interacts_reverse')

        user_updated = self.user_update_fn(user_agg + user_initial)
        user_emb = self.user_proj(user_updated)
        
        dish_agg = tfgnn.keras.layers.SimpleConv(
            sender_node_feature='hidden_state',
            message_fn=self.dish_message_fn,
            reduce_type='sum'
        )(graph, edge_set_name='interacts')

        dish_updated = self.dish_update_fn(dish_agg + dish_initial)
        dish_emb = self.dish_proj(dish_updated)
        
        return user_emb, dish_emb

# if __name__ == "__main__":
#     # 清理旧数据（模拟重新训练）
#     import shutil
#     if os.path.exists("saved_data"):
#         shutil.rmtree("saved_data")
#     if os.path.exists("saved_model"):
#         shutil.rmtree("saved_model")

#     # 初始化并训练
#     recommender = GNNRecommender()
#     recommender.initialize()