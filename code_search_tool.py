import os
import argparse
import torch
import numpy as np
from core import Algo2Vec
from train_repository import FunctionVisitor
from tqdm import tqdm
import sys
import colorama
from colorama import Fore, Style
import platform

# 初始化colorama
colorama.init()

class CodeSearchTool:
    """代码搜索工具"""
    
    def __init__(self, model_path, repo_path=None):
        """
        初始化代码搜索工具
        
        Args:
            model_path: 模型文件路径
            repo_path: 代码库路径（可选）
        """
        self.model_path = model_path
        self.repo_path = repo_path
        self.algo2vec = None
        self.repository_functions = []
        self.function_vectors = []
    
    def load_model(self):
        """加载训练好的algo2vec模型"""
        print(f"加载模型: {self.model_path}")
        try:
            self.algo2vec = Algo2Vec()
            self.algo2vec.load_model(self.model_path)
            print("模型加载成功!")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def index_repository(self, repo_path=None):
        """
        索引代码库中的所有函数
        
        Args:
            repo_path: 代码库路径，如果为None则使用构造函数中的路径
        """
        if repo_path:
            self.repo_path = repo_path
        
        if not self.repo_path:
            print("错误: 未指定代码库路径")
            return False
        
        print(f"开始索引代码库: {self.repo_path}")
        
        # 查找所有Python文件
        python_files = []
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        print(f"找到 {len(python_files)} 个Python文件")
        
        # 提取所有函数
        visitor = FunctionVisitor()
        for py_file in tqdm(python_files):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # 保存文件路径信息
                file_path_rel = os.path.relpath(py_file, self.repo_path)
                
                # 提取函数
                functions = visitor.extract_functions(source_code, py_file)
                
                # 添加文件路径信息
                self.repository_functions.extend([(name, code, file_path_rel) for name, code in functions])
            except Exception as e:
                print(f"处理文件出错: {py_file} - {e}")
        
        print(f"从代码库中提取了 {len(self.repository_functions)} 个函数")
        
        # 构建函数向量
        if self.algo2vec:
            print("为所有函数构建向量表示...")
            self.function_vectors = []
            
            for name, code, path in tqdm(self.repository_functions):
                try:
                    vector = self.algo2vec.get_code_vector(code)
                    self.function_vectors.append(vector)
                except Exception as e:
                    print(f"函数向量化失败: {name} - {e}")
                    # 使用零向量代替失败的向量
                    self.function_vectors.append(np.zeros(self.algo2vec.hidden_dim))
            
            print("函数向量化完成!")
            return True
        else:
            print("错误: 模型未加载")
            return False
    
    def search(self, query_code, top_k=10):
        """
        搜索与查询代码最相似的函数
        
        Args:
            query_code: 查询代码
            top_k: 返回的最相似结果数量
        
        Returns:
            最相似函数的列表，每项包含(相似度, 函数名, 代码, 文件路径)
        """
        if not self.algo2vec:
            print("错误: 模型未加载")
            return []
        
        if not self.repository_functions or not self.function_vectors:
            print("错误: 代码库未索引")
            return []
        
        print("正在搜索相似代码...")
        
        # 获取查询代码的向量表示
        try:
            query_vector = self.algo2vec.get_code_vector(query_code)
        except Exception as e:
            print(f"查询代码向量化失败: {e}")
            return []
        
        # 计算与所有存储函数的相似度
        similarities = []
        for i, (name, code, path) in enumerate(self.repository_functions):
            func_vector = self.function_vectors[i]
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_vector, func_vector)
            similarities.append((similarity, name, code, path))
        
        # 按相似度排序
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # 返回前top_k个结果
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def save_index(self, output_path):
        """保存索引到文件"""
        if not self.repository_functions or not self.function_vectors:
            print("错误: 没有要保存的索引数据")
            return False
        
        print(f"保存索引到: {output_path}")
        
        try:
            # 保存索引数据
            index_data = {
                'repository_path': self.repo_path,
                'functions': [(name, code, path) for name, code, path in self.repository_functions],
                'vectors': np.array(self.function_vectors)
            }
            
            torch.save(index_data, output_path)
            print(f"索引已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"保存索引失败: {e}")
            return False
    
    def load_index(self, index_path):
        """从文件加载索引"""
        print(f"加载索引: {index_path}")
        
        try:
            # 加载索引数据
            index_data = torch.load(index_path)
            
            self.repo_path = index_data['repository_path']
            self.repository_functions = index_data['functions']
            self.function_vectors = index_data['vectors']
            
            print(f"索引加载成功! 包含 {len(self.repository_functions)} 个函数")
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False
    
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
        if not self.algo2vec:
            print("错误: 模型未加载")
            return
        
        if not self.repository_functions or not self.function_vectors:
            print("错误: 代码库未索引")
            return
        
        print("\n===== Algo2Vec 代码搜索 =====")
        print(f"代码库: {self.repo_path}")
        print(f"索引函数数量: {len(self.repository_functions)}")
        print("输入 'q' 或 'quit' 退出")
        print("============================\n")
        
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
            
            for i, (similarity, name, code, path) in enumerate(results):
                print(f"\n{Fore.CYAN}[{i+1}] {name} ({similarity:.2f}){Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}文件: {path}{Style.RESET_ALL}")
                
                # 显示代码片段
                highlighted_code = self.highlight_code(code)
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
    parser = argparse.ArgumentParser(description='Algo2Vec代码搜索工具')
    parser.add_argument('--model', required=True, help='训练好的模型路径')
    parser.add_argument('--repo', help='要索引的代码库路径')
    parser.add_argument('--index', help='索引文件路径，用于保存或加载索引')
    parser.add_argument('--query', help='查询代码文件路径')
    parser.add_argument('--save-index', action='store_true', help='保存索引到文件')
    parser.add_argument('--load-index', action='store_true', help='从文件加载索引')
    parser.add_argument('--top-k', type=int, default=10, help='返回的最相似结果数量')
    parser.add_argument('--interactive', action='store_true', help='启动交互式搜索界面')
    
    args = parser.parse_args()
    
    # 创建搜索工具
    search_tool = CodeSearchTool(args.model, args.repo)
    
    # 加载模型
    if not search_tool.load_model():
        return
    
    # 加载索引
    if args.load_index and args.index:
        search_tool.load_index(args.index)
    # 或者索引代码库
    elif args.repo:
        search_tool.index_repository()
        # 保存索引
        if args.save_index and args.index:
            search_tool.save_index(args.index)
    
    # 交互式搜索
    if args.interactive:
        search_tool.interactive_search()
    # 查询文件
    elif args.query:
        try:
            with open(args.query, 'r', encoding='utf-8') as f:
                query_code = f.read()
            
            results = search_tool.search(query_code, args.top_k)
            
            # 显示结果
            print(f"\n找到 {len(results)} 个相似函数:")
            
            for i, (similarity, name, code, path) in enumerate(results):
                print(f"\n[{i+1}] {name} ({similarity:.2f})")
                print(f"文件: {path}")
                print(code[:200] + "..." if len(code) > 200 else code)
        except Exception as e:
            print(f"查询失败: {e}")

if __name__ == "__main__":
    main()