import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import argparse
import faiss
import numpy as np
import torch
from tqdm import tqdm
import pickle
import time
import colorama
from colorama import Fore, Style

# 导入algo2vec相关模块
from core import Algo2Vec
from train_repository import FunctionVisitor

# 初始化colorama
colorama.init()


class CodeVectorizer:
    """使用algo2vec将代码转换为向量表示"""
    
    def __init__(self, model_path):
        """初始化模型"""
        self.algo2vec = Algo2Vec()
        print(f"加载模型: {model_path}")
        try:
            self.algo2vec.load_model(model_path)
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
            
    def vectorize(self, code):
        """将代码转换为向量"""
        return self.algo2vec.get_code_vector(code)
    
    @property
    def vector_dim(self):
        """获取向量维度"""
        return self.algo2vec.hidden_dim


class VectorRepository:
    """使用FAISS管理代码向量存储和检索"""
    
    def __init__(self, dimension, index_type='flat'):
        """初始化向量库"""
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index(index_type, dimension)
        self.metadata = []  # 存储与向量对应的元数据（函数名、路径等）
        
    def _create_index(self, index_type, dimension):
        """创建FAISS索引"""
        if index_type == 'flat':
            # 最基础的精确搜索，适合小规模数据集（<100k向量）
            return faiss.IndexFlatL2(dimension)
        elif index_type == 'ivf':
            # IVF索引，适合中等规模数据集
            n_list = max(int(np.sqrt(1000)), 1)  # 聚类数，通常为训练样本数的平方根
            quantizer = faiss.IndexFlatL2(dimension)
            return faiss.IndexIVFFlat(quantizer, dimension, n_list)
        elif index_type == 'hnsw':
            # HNSW索引，适合大规模数据集，性能最佳
            return faiss.IndexHNSWFlat(dimension, 32)  # 32是每个节点的连接数
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
    
    def add_vectors(self, vectors, metadata_list):
        """添加向量及其元数据到索引"""
        if len(vectors) == 0:
            return
            
        vectors_np = np.array(vectors).astype('float32')
        
        # 对于某些索引类型，可能需要先训练
        if self.index_type == 'ivf' and not self.index.is_trained:
            print("训练IVF索引...")
            self.index.train(vectors_np)
            
        # 添加向量到索引
        self.index.add(vectors_np)
        
        # 存储元数据
        self.metadata.extend(metadata_list)
        
        print(f"已添加 {len(vectors)} 个向量到索引，总计 {len(self.metadata)} 个")
        
    def search(self, query_vector, k=5):
        """搜索最相似的k个向量"""
        query_np = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):  # -1表示没有找到足够的结果
                results.append((self.metadata[idx], float(distances[0][i])))
                
        return results
    
    def save(self, index_path, metadata_path):
        """保存索引和元数据"""
        # 保存FAISS索引
        faiss.write_index(self.index, index_path)
        
        # 保存元数据
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"索引已保存到 {index_path}")
        print(f"元数据已保存到 {metadata_path}")
    
    @classmethod
    def load(cls, index_path, metadata_path):
        """加载索引和元数据"""
        # 加载FAISS索引
        index = faiss.read_index(index_path)
        
        # 加载元数据
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 创建实例并设置属性
        instance = cls(index.d)  # 使用索引的维度初始化
        instance.index = index
        instance.metadata = metadata
        
        print(f"从 {index_path} 加载了索引")
        print(f"从 {metadata_path} 加载了 {len(metadata)} 个元数据")
        
        return instance


class CodeSearchEngine:
    """代码搜索引擎，集成代码向量化和向量检索功能"""
    
    def __init__(self, model_path=None, index_path=None, metadata_path=None, index_type='flat'):
        """初始化搜索引擎"""
        # 如果指定了模型路径，加载代码向量化组件
        if model_path:
            self.vectorizer = CodeVectorizer(model_path)
            
            # 如果指定了索引路径，加载现有索引；否则创建新索引
            if index_path and metadata_path and os.path.exists(index_path) and os.path.exists(metadata_path):
                self.repository = VectorRepository.load(index_path, metadata_path)
            else:
                self.repository = VectorRepository(self.vectorizer.vector_dim, index_type)
        else:
            raise ValueError("必须提供模型路径")
    
    def index_repository(self, repo_path, save_paths=None):
        """索引代码仓库"""
        print(f"开始索引代码仓库: {repo_path}")
        
        # 查找所有Python文件
        python_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        print(f"找到 {len(python_files)} 个Python文件")
        
        # 提取所有函数
        visitor = FunctionVisitor()
        all_functions = []
        file_paths = []
        
        for py_file in tqdm(python_files, desc="提取函数"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # 提取函数
                functions = visitor.extract_functions(source_code, py_file)
                all_functions.extend(functions)
                
                # 记录文件路径
                rel_path = os.path.relpath(py_file, repo_path)
                file_paths.extend([rel_path] * len(functions))
            except Exception as e:
                print(f"处理文件出错: {py_file} - {e}")
        
        print(f"共提取了 {len(all_functions)} 个函数")
        
        # 向量化并添加到索引
        batch_size = 100  # 批处理大小
        for i in tqdm(range(0, len(all_functions), batch_size), desc="向量化"):
            batch_functions = all_functions[i:i+batch_size]
            batch_paths = file_paths[i:i+batch_size]
            
            vectors = []
            metadata = []
            
            for j, (name, code) in enumerate(batch_functions):
                try:
                    vector = self.vectorizer.vectorize(code)
                    vectors.append(vector)
                    
                    # 创建元数据
                    metadata.append({
                        'name': name,
                        'path': batch_paths[j],
                        'code': code
                    })
                except Exception as e:
                    print(f"向量化失败: {name} - {e}")
            
            # 添加到索引
            if vectors:
                self.repository.add_vectors(vectors, metadata)
        
        # 保存索引
        if save_paths:
            index_path, metadata_path = save_paths
            self.repository.save(index_path, metadata_path)
        
        return len(all_functions)
    
    def search(self, query_code, top_k=5):
        """搜索相似代码"""
        try:
            # 向量化查询代码
            query_vector = self.vectorizer.vectorize(query_code)
            
            # 搜索相似向量
            start_time = time.time()
            results = self.repository.search(query_vector, top_k)
            search_time = time.time() - start_time
            
            print(f"搜索完成，用时 {search_time:.4f} 秒")
            return results
        except Exception as e:
            print(f"搜索出错: {e}")
            return []
    
    def highlight_code(self, code):
        """对代码进行简单的语法高亮"""
        # 替换缩进
        code = code.replace("    ", "  ")
        
        # 关键字高亮
        keywords = ["def", "if", "else", "elif", "for", "while", "try", "except", 
                   "finally", "with", "return", "class", "import", "from", "as", 
                   "break", "continue", "pass", "None", "True", "False"]
        
        for keyword in keywords:
            code = code.replace(f" {keyword} ", f" {Fore.BLUE}{keyword}{Style.RESET_ALL} ")
            code = code.replace(f" {keyword}:", f" {Fore.BLUE}{keyword}{Style.RESET_ALL}:")
            code = code.replace(f"\n{keyword} ", f"\n{Fore.BLUE}{keyword}{Style.RESET_ALL} ")
            code = code.replace(f"\n{keyword}:", f"\n{Fore.BLUE}{keyword}{Style.RESET_ALL}:")
        
        # 字符串高亮
        lines = code.split("\n")
        highlighted_lines = []
        for line in lines:
            # 简单处理单引号和双引号字符串
            in_single_quote = False
            in_double_quote = False
            result = ""
            
            for char in line:
                if char == "'" and not in_double_quote:
                    if in_single_quote:
                        result += f"{Style.RESET_ALL}"
                        in_single_quote = False
                    else:
                        result += f"{Fore.GREEN}"
                        in_single_quote = True
                elif char == '"' and not in_single_quote:
                    if in_double_quote:
                        result += f"{Style.RESET_ALL}"
                        in_double_quote = False
                    else:
                        result += f"{Fore.GREEN}"
                        in_double_quote = True
                
                result += char
            
            highlighted_lines.append(result)
        
        return "\n".join(highlighted_lines)
    
    def interactive_search(self):
        """交互式搜索界面"""
        print(f"\n{Fore.CYAN}===== Algo2Vec+FAISS 代码搜索 ====={Style.RESET_ALL}")
        print(f"索引大小: {len(self.repository.metadata)} 个函数")
        print("输入 'q' 或 'quit' 退出")
        print(f"{Fore.CYAN}===================================={Style.RESET_ALL}\n")
        
        while True:
            # 获取用户输入
            print(f"{Fore.YELLOW}请输入查询代码 (多行输入，输入空行结束):{Style.RESET_ALL}")
            query_lines = []
            
            while True:
                line = input()
                if not line.strip():
                    break
                if line.strip().lower() in ('q', 'quit', 'exit'):
                    return
                query_lines.append(line)
            
            query_code = "\n".join(query_lines)
            
            if not query_code.strip():
                print("查询代码为空，请重新输入")
                continue
            
            # 查询
            top_k = 5
            try:
                top_k = int(input(f"{Fore.YELLOW}返回结果数量 [{top_k}]: {Style.RESET_ALL}") or top_k)
            except ValueError:
                print("输入无效，使用默认值")
            
            # 执行搜索
            results = self.search(query_code, top_k=top_k)
            
            if not results:
                print(f"{Fore.RED}未找到相似代码{Style.RESET_ALL}")
                continue
            
            # 显示结果
            print(f"\n{Fore.GREEN}找到 {len(results)} 个相似函数:{Style.RESET_ALL}")
            
            for i, (metadata, distance) in enumerate(results):
                similarity = 1.0 / (1.0 + distance)  # 转换距离为相似度
                print(f"\n{Fore.CYAN}[{i+1}] {metadata['name']} (相似度: {similarity:.2f}){Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}文件: {metadata['path']}{Style.RESET_ALL}")
                
                # 显示代码片段
                highlighted_code = self.highlight_code(metadata['code'])
                print(f"{highlighted_code}\n")
                
                # 每次显示3个结果后暂停
                if (i + 1) % 3 == 0 and i + 1 < len(results):
                    if input("继续显示更多结果? (y/n): ").lower() != 'y':
                        break
            
            # 询问是否继续搜索
            if input(f"\n{Fore.YELLOW}继续搜索? (y/n): {Style.RESET_ALL}").lower() != 'y':
                break
        
        print("搜索结束")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Algo2Vec+FAISS代码搜索工具')
    parser.add_argument('--model', required=True, help='algo2vec模型路径')
    parser.add_argument('--repo', help='要索引的代码库路径')
    parser.add_argument('--index', help='FAISS索引文件路径')
    parser.add_argument('--metadata', help='与索引配套的元数据文件路径')
    parser.add_argument('--index-type', choices=['flat', 'ivf', 'hnsw'], default='flat', 
                      help='FAISS索引类型: flat(精确搜索), ivf(快速近似), hnsw(高性能图索引)')
    parser.add_argument('--save-index', action='store_true', help='保存索引到文件')
    parser.add_argument('--query', help='查询代码文件路径')
    parser.add_argument('--top-k', type=int, default=5, help='返回的最相似结果数量')
    parser.add_argument('--interactive', action='store_true', help='启动交互式搜索界面')
    
    args = parser.parse_args()
    
    # 检查FAISS GPU支持
    gpu_available = torch.cuda.is_available()
    # gpu_available = False
    try:
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            print(f"FAISS检测到 {faiss.get_num_gpus()} 个GPU，将启用GPU加速")
            gpu_available = True
    except:
        pass
        
    if not gpu_available:
        print("FAISS未检测到GPU，使用CPU模式")
    
    # 设置索引和元数据文件路径
    index_path = args.index
    metadata_path = args.metadata
    
    if args.save_index:
        if not index_path:
            index_path = "code_vectors.index"
        if not metadata_path:
            metadata_path = "code_metadata.pkl"
    
    # 初始化搜索引擎
    try:
        engine = CodeSearchEngine(
            model_path=args.model,
            index_path=index_path,
            metadata_path=metadata_path,
            index_type=args.index_type
        )
    except Exception as e:
        print(f"初始化搜索引擎失败: {e}")
        return
    
    # 索引代码库
    if args.repo:
        save_paths = (index_path, metadata_path) if args.save_index else None
        engine.index_repository(args.repo, save_paths)
    
    # 交互式搜索
    if args.interactive:
        engine.interactive_search()
    # 查询文件
    elif args.query:
        try:
            with open(args.query, 'r', encoding='utf-8') as f:
                query_code = f.read()
            
            results = engine.search(query_code, args.top_k)
            
            # 显示结果
            print(f"\n找到 {len(results)} 个相似函数:")
            
            for i, (metadata, distance) in enumerate(results):
                similarity = 1.0 / (1.0 + distance)  # 转换距离为相似度
                print(f"\n[{i+1}] {metadata['name']} (相似度: {similarity:.2f})")
                print(f"文件: {metadata['path']}")
                print(metadata['code'][:200] + "..." if len(metadata['code']) > 200 else metadata['code'])
        except Exception as e:
            print(f"查询失败: {e}")


if __name__ == "__main__":
    main()