from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import Docx2txtLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
class RAG:
    def __init__(self):
      self=None

    # 加载pdf文件
    def loadpdf(self, path):
       return PyPDFLoader(path).load()
    # 加载txt文件
    def loadtxt(self, path):
       return TextLoader(path).load()
    # 加载md文件
    def loadmd(self, path):
       return UnstructuredMarkdownLoader(path).load()
    # 加载docx文件
    def loaddocx(self, path):
       return Docx2txtLoader(path).load()
    
    # 给定具体文件路径 加载文件
    def load(self, path):
        if path.endswith('.pdf'):
            return self.loadpdf(path)
        elif path.endswith('.txt'):
            return self.loadtxt(path)
        elif path.endswith('.md'):
            return self.loadmd(path)
        elif path.endswith('.docx'):
            return self.loaddocx(path)
        else:
            raise ValueError('Unsupported file format')

    # 获取文件夹下所有文件路径  作为列表返回
    def get_paths(self, dir):
        paths=[] 
        for i in os.listdir(dir):
            full_path = os.path.join(dir, i)
            if os.path.isdir(full_path):
                paths.extend(self.get_paths(full_path))
            else:
                paths.append(full_path)
        return paths


# 加载文件夹下所有文件，并输出错误信息   得到documents列表
    def load_all(self, dir):
        if not os.path.isdir(dir):
            print(f"错误: {dir} 不是一个有效的目录")
            return []

        paths = self.get_paths(dir)
        all_documents = []  # 存储成功加载的文档
        print(all_documents)
        for path in paths:
            try:
                print(f"正在加载: {path}")
                documents = self.load(path)  # 加载文件
                all_documents.extend(documents)
                print(f"成功加载: {path}，文档数: {len(documents)}")
            except FileNotFoundError as e:
                print(f"加载失败: {path} - 文件不存在 ({str(e)})")
            except ValueError as e:
                print(f"加载失败: {path} - {str(e)}")  # 包括空文件或格式不支持
            except Exception as e:
                print(f"加载失败: {path} - 未知错误 ({str(e)})")
            print('\n')
        
        print(f"\n总共成功加载的文档数量: {len(all_documents)}")
        return all_documents
    
    #文档切割      对documents进行切割
    def split_documents(self, documents, chunk_size=500, chunk_overlap=200):
        """
        对加载好的文档进行切割。

        参数:
        - documents: 加载好的文档列表，每个元素是一个 Document 对象
        - chunk_size: 每个文本块的最大字符数（默认 500）
        - chunk_overlap: 文本块之间的重叠字符数（默认 200）

        返回:
        - 分割后的文档列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,    # 每个块的大小
            chunk_overlap=chunk_overlap,  # 块之间的重叠
            length_function=len,      # 计算长度的函数（默认字符数）
            add_start_index=True      # 可选：保留原始文档中的起始位置
        )

        split_docs = text_splitter.split_documents(documents)
        return split_docs
    

# 生成嵌入并存储到向量数据库
    def embed_documents(self, documents, embedding_model="smartcreation/bge-large-zh-v1.5:latest", persist_directory="./chroma_db"):
        """
        对文档生成嵌入并存储。
        
        参数:
        - documents: 要嵌入的文档列表（默认使用 self.split_docs）
        - embedding_model: 使用的嵌入模型名称
        - persist_directory: 向量数据库存储路径
        
        返回:
        - Chroma 向量存储对象
        """
        # 初始化嵌入模型
        print(f"正在初始化嵌入模型: {embedding_model}")
        embeddings = OllamaEmbeddings(model=embedding_model)

        # 创建向量存储并嵌入文档
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory  # 可选：持久化存储
        )
        
        print(f"已生成嵌入并存储，文档数量: {len(documents)}")
        return vector_store

    def load_vector_store(self, embedding_model="smartcreation/bge-large-zh-v1.5:latest", persist_directory="./chroma_db"):
        """从持久化目录加载现有的向量数据库"""
        if not os.path.exists(persist_directory):
            raise ValueError(f"向量数据库目录 {persist_directory} 不存在，请先嵌入文档")
        
        print(f"正在加载向量数据库从: {persist_directory}")
        embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        print("向量数据库加载成功")
        return self.vector_store
# 根据查询检索相关文档
    def retrieve(self,vector_store, query, top_k=3):
        """
        从知识库中检索与查询最相关的文档。
        
        参数:
        - query: 查询文本（字符串）
        - top_k: 返回的最相关文档数量（默认 3）
        
        返回:
        - List[Document]: 最相关的文档列表
        """
        if vector_store is None:
            raise ValueError("向量知识库未初始化，请先建立向量知识库")
        
        # 创建检索器
        retriever = vector_store.as_retriever(
            search_type="similarity",  # 基于相似性检索
            search_kwargs={"k": top_k}  # 返回 top_k 个结果
        )
        
        # 执行检索
        results = retriever.get_relevant_documents(query)
        return results
    # 对检索到的文档进行过滤：如果文档的相似度分数低于设定的阈值，则不返回该文档
    
















