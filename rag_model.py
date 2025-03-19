import os
from typing import List, Dict
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.vectorstores import Chroma
from huggingface_hub import snapshot_download
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM_methods:
    def __init__(self):
        self.llm_path='models\deepseek-ai\DeepSeek-R1-Distill-Qwen-1.5B'
        self.embeddingname='paraphrase-multilingual-MiniLM-L12-v2'
        self.embedding=None
        self.llm=None
        self.tokens=None



# 下载文本生成模型至本地 传入embedding模型名称和llm模型名称 
    def download_model(embeddingname,LLM_model: str,self):
        self.embeddingname=embeddingname
        """
    下载模型到本地models目录
    Args:
        LLM_model: Hugging Face LLM模型名称
        """
    # 创建models目录
        os.makedirs("models", exist_ok=True)
    
    # 下载LLM模型
        llm_path = f"models/{LLM_model}"
        if not os.path.exists(llm_path):
            snapshot_download(
            repo_id=LLM_model,
            local_dir=llm_path,
            local_dir_use_symlinks=False  # 不使用符号链接，直接复制文件
            )
            print(f"模型已成功下载到: {llm_path}")
            self.llm_path=llm_path
        else :
            print(f"模型已存在: {llm_path}")
            self.llm_path=llm_path
            return 

# 获取本地模型 初始化本地模型 包括embedding模型和llm模型以及llm分词器
    def prepare(self):
        # 初始化 embedding 模型
        embedding_model= HuggingFaceEmbeddings(
        model_name=self.embeddingname,
        )
        print("embedding 已从huggingface获取") 
        self.embedding=embedding_model
        
        # 初始化 LLM 模型
        # 加载分词器和生成模型
        llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_path)
        print("分词器加载完成")
        self.tokens=llm_tokenizer
        llm_model = AutoModelForCausalLM.from_pretrained(self.llm_path)
        print("生成模型加载完成")
        self.llm=llm_model
        return embedding_model,llm_model,llm_tokenizer
        
    def generate_text(self,input_text):
        # 将模型移动到 GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.to(device)
        # 准备输入数据
        inputs = self.tokens(input_text, return_tensors="pt").to(device)
        # 进行推理
        with torch.no_grad():
            outputs =self.llm.generate(**inputs, max_length=50)
        # 处理输出数据
        generated_text = self.tokens.decode(outputs[0], skip_special_tokens=True)
        return generated_text






