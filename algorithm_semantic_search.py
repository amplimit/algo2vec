import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import ast
import re
import faiss
import numpy as np
from collections import Counter
import pickle
from tqdm import tqdm

# 假设已有algo2vec和原有代码
from core import Algo2Vec, PathExtractor

class AlgorithmFeatureExtractor:
    """提取代码中的算法语义特征"""
    
    # 算法领域特定的特征
    ALGORITHM_FEATURES = {
        'arithmetic': ['%', '//', '/', '*', '+', '-', '**', 'math.'],
        'prime': ['%', 'prime', 'divisor', 'factor'],
        'sort': ['sort', 'swap', 'bubble', 'quick', 'merge', '<', '>'],
        'search': ['find', 'search', 'binary', 'linear', 'index', 'locate'],
        'graph': ['node', 'edge', 'vertex', 'graph', 'adjacency', 'dfs', 'bfs'],
        'tree': ['root', 'leaf', 'node', 'tree', 'binary', 'traverse', 'depth'],
        'dynamic': ['dp', 'memo', 'dynamic', 'programming', 'fibonacci', 'subproblem'],
        'string': ['substring', 'pattern', 'match', 'char', 'text', 'str']
    }
    
    def __init__(self):
        """初始化"""
        self.algorithm_patterns = {}
        for algo_type, keywords in self.ALGORITHM_FEATURES.items():
            # 转义特殊字符，避免正则表达式错误
            escaped_keywords = [re.escape(keyword) for keyword in keywords]
            self.algorithm_patterns[algo_type] = re.compile('|'.join(escaped_keywords), re.IGNORECASE)
    
    def extract_features(self, code):
        """提取代码的算法特征"""
        # 初始化默认特征值，确保所有键都存在
        features = {
            'loops': 0,
            'if_conditions': 0,
            'calls': 0,
            'returns': 0,
            'assignments': 0,
            'comparisons': 0,
            'nesting_depth': 0,
            'avg_call_args': 0,
            'operator_types': 0,
            'variable_names': [],
            'function_names': [],
            'has_small_primes': False,
            'has_mod_check': False,
            'has_sqrt_bound': False,
            'has_range_loop': False,
            'has_comparison_in_loop': False,
            'algorithm_scores': {k: 0.0 for k in self.ALGORITHM_FEATURES}
        }
        
        # 基础结构特征
        try:
            tree = ast.parse(code)
            
            # 计数各种节点
            node_counts = Counter(type(node).__name__ for node in ast.walk(tree))
            
            # 提取特定结构特征
            features['loops'] = node_counts.get('For', 0) + node_counts.get('While', 0)
            features['if_conditions'] = node_counts.get('If', 0)
            features['calls'] = node_counts.get('Call', 0)
            features['returns'] = node_counts.get('Return', 0)
            features['assignments'] = node_counts.get('Assign', 0) + node_counts.get('AugAssign', 0)
            features['comparisons'] = node_counts.get('Compare', 0)
            
            # 复杂度特征
            features['nesting_depth'] = self._calc_nesting_depth(tree)
            features['avg_call_args'] = self._avg_call_args(tree)
            
            # 操作符特征
            features['operator_types'] = len(set(node.op.__class__.__name__ 
                                                for node in ast.walk(tree) 
                                                if hasattr(node, 'op')))
            
            # 变量使用特征
            features['variable_names'] = list(set(node.id for node in ast.walk(tree) 
                                                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)))
            
            # 获取所有字面常数
            literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
            features['has_small_primes'] = any(val in [2, 3, 5, 7, 11, 13, 17, 19] for val in literals if isinstance(val, int))
            
            # 获取函数名称（从FunctionDef和之后的Name节点）
            func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            features['function_names'] = func_names
            
        except SyntaxError:
            # 如果AST解析失败，使用简单的文本特征
            features['loops'] = code.count('for ') + code.count('while ')
            features['if_conditions'] = code.count('if ') - code.count('elif ') 
            features['returns'] = code.count('return ')
            features['function_names'] = []
        
        # 提取算法领域特征
        algorithm_scores = {}
        for algo_type, pattern in self.algorithm_patterns.items():
            matches = pattern.findall(code)
            algorithm_scores[algo_type] = len(matches) / max(len(code.split()), 1)
        
        features['algorithm_scores'] = algorithm_scores
        
        # 提取特定的算法结构模式
        features['has_mod_check'] = '%' in code and 'return False' in code
        features['has_sqrt_bound'] = 'sqrt' in code or '**0.5' in code
        features['has_range_loop'] = 'range(' in code
        features['has_comparison_in_loop'] = bool(re.search(r'for.*:.*if.*[<>=!]', code, re.DOTALL))
        
        return features
    
    def _calc_nesting_depth(self, tree):
        """计算代码的嵌套深度"""
        max_depth = 0
        current_depth = 0
        
        class DepthVisitor(ast.NodeVisitor):
            def visit_For(self, node):
                nonlocal current_depth, max_depth
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                self.generic_visit(node)
                current_depth -= 1
                
            def visit_While(self, node):
                nonlocal current_depth, max_depth
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                self.generic_visit(node)
                current_depth -= 1
                
            def visit_If(self, node):
                nonlocal current_depth, max_depth
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                self.generic_visit(node)
                current_depth -= 1
        
        DepthVisitor().visit(tree)
        return max_depth
    
    def _avg_call_args(self, tree):
        """计算函数调用的平均参数数量"""
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        if not calls:
            return 0
        arg_counts = [len(call.args) for call in calls]
        return sum(arg_counts) / len(arg_counts)
    
    def features_to_vector(self, features):
        """将提取的特征转换为向量"""
        # 结构特征 - 使用get方法安全访问
        structure_vector = np.array([
            features.get('loops', 0),
            features.get('if_conditions', 0),
            features.get('calls', 0),
            features.get('returns', 0),
            features.get('assignments', 0),
            features.get('comparisons', 0),
            features.get('nesting_depth', 0),
            features.get('avg_call_args', 0),
            features.get('operator_types', 0),
            int(features.get('has_mod_check', False)),
            int(features.get('has_sqrt_bound', False)),
            int(features.get('has_range_loop', False)),
            int(features.get('has_comparison_in_loop', False)),
            int(features.get('has_small_primes', False))
        ], dtype=np.float32)
        
        # 归一化结构向量
        structure_max = np.array([5, 10, 20, 5, 20, 10, 5, 5, 10, 1, 1, 1, 1, 1], dtype=np.float32)
        structure_vector = np.clip(structure_vector / structure_max, 0, 1)
        
        # 算法类型向量
        algorithm_vector = np.array([
            features.get('algorithm_scores', {}).get('arithmetic', 0),
            features.get('algorithm_scores', {}).get('prime', 0),
            features.get('algorithm_scores', {}).get('sort', 0),
            features.get('algorithm_scores', {}).get('search', 0),
            features.get('algorithm_scores', {}).get('graph', 0),
            features.get('algorithm_scores', {}).get('tree', 0),
            features.get('algorithm_scores', {}).get('dynamic', 0),
            features.get('algorithm_scores', {}).get('string', 0)
        ], dtype=np.float32)
        
        # 合并向量
        return np.concatenate([structure_vector, algorithm_vector])
    
    def get_algorithm_type(self, features):
        """根据特征判断代码可能实现的算法类型"""
        algorithm_scores = features['algorithm_scores']
        algo_type = max(algorithm_scores.items(), key=lambda x: x[1], default=('unknown', 0))
        
        # 特殊处理素数检查算法
        if (features['has_mod_check'] and features['has_range_loop'] and 
            features['algorithm_scores'].get('prime', 0) > 0):
            return 'prime_check'
        
        # 特殊处理排序算法
        if (features['loops'] >= 2 and features['has_comparison_in_loop'] and 
            features['algorithm_scores'].get('sort', 0) > 0):
            return 'sorting'
        
        # 返回得分最高的算法类型
        if algo_type[1] > 0.1:  # 设置一个阈值
            return algo_type[0]
        else:
            return 'general'


class EnhancedCodeVectorizer:
    """结合algo2vec和算法特征的增强代码向量化"""
    
    def __init__(self, algo2vec_model):
        """初始化"""
        self.algo2vec_model = algo2vec_model
        self.feature_extractor = AlgorithmFeatureExtractor()
        self.vector_dim = algo2vec_model.hidden_dim + 22  # 基础向量维度 + 特征向量维度
    
    def vectorize(self, code):
        """将代码转换为增强向量"""
        # 获取algo2vec向量
        algo2vec_vector = self.algo2vec_model.get_code_vector(code)
        
        # 提取算法特征
        features = self.feature_extractor.extract_features(code)
        
        # 将特征转换为向量
        feature_vector = self.feature_extractor.features_to_vector(features)
        
        # 混合向量 - 算法特征占25%权重
        algo2vec_weight = 0.75
        feature_weight = 0.25
        
        # 调整向量长度匹配
        algo2vec_vector_norm = algo2vec_vector / np.linalg.norm(algo2vec_vector)
        feature_vector_norm = feature_vector / np.linalg.norm(feature_vector)
        
        # 扩展特征向量到algo2vec向量的维度
        extended_feature_vector = np.zeros_like(algo2vec_vector)
        for i in range(len(feature_vector)):
            extended_feature_vector[i % len(algo2vec_vector)] += feature_vector[i]
        extended_feature_vector = extended_feature_vector / np.linalg.norm(extended_feature_vector)
        
        # 融合向量
        combined_vector = (algo2vec_vector_norm * algo2vec_weight) + (extended_feature_vector * feature_weight)
        
        # 返回归一化后的向量
        return combined_vector / np.linalg.norm(combined_vector)
    
    def analyze_code(self, code):
        """分析代码并返回算法类型和特征"""
        features = self.feature_extractor.extract_features(code)
        algorithm_type = self.feature_extractor.get_algorithm_type(features)
        
        return {
            'algorithm_type': algorithm_type,
            'features': features
        }


class AlgorithmSemanticSearch:
    """基于算法语义的代码搜索"""
    
    def __init__(self, algo2vec_model_path):
        """初始化"""
        # 加载algo2vec模型
        self.algo2vec = Algo2Vec()
        print(f"加载algo2vec模型: {algo2vec_model_path}")
        self.algo2vec.load_model(algo2vec_model_path)
        
        # 创建增强向量器
        self.vectorizer = EnhancedCodeVectorizer(self.algo2vec)
        
        # 创建FAISS索引
        self.index = None
        self.metadata = []
    
    def create_index(self, index_type='flat'):
        """创建FAISS索引"""
        dimension = self.vectorizer.vector_dim
        
        if index_type == 'flat':
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)
        elif index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
        
        print(f"已创建{index_type}类型索引，维度:{dimension}")
    
    def index_repository(self, repo_path, recursive=True):
        """索引代码仓库"""
        if self.index is None:
            self.create_index()
        
        print(f"开始索引代码仓库: {repo_path}")
        
        # 查找所有Python文件
        python_files = []
        if recursive:
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
        else:
            python_files = [os.path.join(repo_path, f) for f in os.listdir(repo_path) 
                           if f.endswith('.py') and os.path.isfile(os.path.join(repo_path, f))]
        
        print(f"找到 {len(python_files)} 个Python文件")
        
        # 提取所有函数
        from train_repository import FunctionVisitor
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
        batch_size = 50  # 减小批次大小，避免内存问题
        vectors = []
        metadata_list = []
        
        for i in tqdm(range(0, len(all_functions)), desc="向量化"):
            name, code = all_functions[i]
            path = file_paths[i]
            
            try:
                # 向量化并分析代码
                vector = self.vectorizer.vectorize(code)
                analysis = self.vectorizer.analyze_code(code)
                
                vectors.append(vector)
                
                # 创建元数据
                metadata_list.append({
                    'name': name,
                    'path': path,
                    'code': code,
                    'algorithm_type': analysis['algorithm_type']
                })
                
                # 批处理添加到索引
                if len(vectors) >= batch_size:
                    self._add_vectors_to_index(vectors, metadata_list)
                    vectors = []
                    metadata_list = []
                    
            except Exception as e:
                print(f"向量化失败: {name} - {e}")
        
        # 添加剩余的向量
        if vectors:
            self._add_vectors_to_index(vectors, metadata_list)
        
        print(f"索引构建完成，共索引了 {len(self.metadata)} 个函数")
        
        # 统计算法类型分布
        algo_types = Counter(meta['algorithm_type'] for meta in self.metadata)
        print("\n算法类型分布:")
        for algo_type, count in algo_types.most_common():
            print(f"  {algo_type}: {count}")
    
    def _add_vectors_to_index(self, vectors, metadata_list):
        """添加向量到索引"""
        if not vectors:
            return
        
        # 将向量转换为numpy数组
        vectors_np = np.array(vectors).astype('float32')
        
        # 检查向量是否包含NaN或无穷大值
        if np.isnan(vectors_np).any() or np.isinf(vectors_np).any():
            print("警告: 向量中包含NaN或无穷大值，已被过滤")
            vectors_np = np.nan_to_num(vectors_np)
        
        # 对于某些索引类型，可能需要先训练
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            if len(vectors_np) < 100:  # IVF需要足够多的训练样本
                # 生成随机样本用于训练
                print("生成随机样本用于IVF索引训练")
                train_vectors = np.random.randn(max(100, len(vectors_np)*2), vectors_np.shape[1]).astype('float32')
                train_vectors = train_vectors / np.linalg.norm(train_vectors, axis=1, keepdims=True)
                self.index.train(train_vectors)
            else:
                print("训练IVF索引...")
                self.index.train(vectors_np)
        
        # 添加向量到索引
        self.index.add(vectors_np)
        
        # 存储元数据
        self.metadata.extend(metadata_list)
    
    def search(self, query_code, top_k=5, filter_algo_type=None):
        """搜索相似代码"""
        if self.index is None or len(self.metadata) == 0:
            print("索引未创建或为空")
            return []
        
        try:
            # 分析查询代码
            analysis = self.vectorizer.analyze_code(query_code)
            query_algo_type = analysis['algorithm_type']
            print(f"查询代码分析结果: 算法类型={query_algo_type}")
            print(f"特征分数: {analysis['features']['algorithm_scores']}")
            
            # 向量化查询代码
            query_vector = self.vectorizer.vectorize(query_code)
            query_vector = query_vector.astype('float32')
            
            # 决定是否应用算法类型过滤
            apply_filter = filter_algo_type or (query_algo_type != 'general')
            
            # 获取更多候选结果以便过滤
            search_k = top_k * 3 if apply_filter else top_k
            
            # 搜索相似向量
            distances, indices = self.index.search(np.array([query_vector]), search_k)
            
            # 收集结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1 or idx >= len(self.metadata):
                    continue
                    
                metadata = self.metadata[idx]
                distance = float(distances[0][i])
                similarity = np.exp(-distance)  # 将距离转换为相似度
                
                results.append((metadata, distance, similarity))
            
            # 按算法类型过滤和重排序
            if apply_filter and results:  # 确保有结果
                # 确定要过滤的类型
                filter_type = filter_algo_type or query_algo_type
                
                # 提升相同算法类型的结果排名
                ranked_results = []
                for metadata, distance, similarity in results:
                    algo_match_bonus = 0.3 if metadata.get('algorithm_type', '') == filter_type else 0
                    adjusted_similarity = similarity + algo_match_bonus
                    ranked_results.append((metadata, distance, adjusted_similarity))
                
                # 重新排序
                ranked_results.sort(key=lambda x: x[2], reverse=True)
                
                # 截取前top_k个
                return ranked_results[:top_k]
            else:
                return results[:top_k]
            
        except Exception as e:
            print(f"搜索出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def save_index(self, index_path, metadata_path):
        """保存索引和元数据"""
        if self.index is None:
            print("索引未创建，无法保存")
            return False
        
        # 保存FAISS索引
        faiss.write_index(self.index, index_path)
        
        # 保存元数据
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"索引已保存到: {index_path}")
        print(f"元数据已保存到: {metadata_path}")
        return True
    
    def load_index(self, index_path, metadata_path):
        """加载索引和元数据"""
        # 加载FAISS索引
        self.index = faiss.read_index(index_path)
        
        # 加载元数据
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"从 {index_path} 加载了索引")
        print(f"从 {metadata_path} 加载了 {len(self.metadata)} 个元数据")
        return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基于算法语义的代码搜索')
    parser.add_argument('--model', required=True, help='algo2vec模型路径')
    parser.add_argument('--repo', help='要索引的代码库路径')
    parser.add_argument('--index', help='FAISS索引文件路径')
    parser.add_argument('--metadata', help='元数据文件路径')
    parser.add_argument('--index-type', choices=['flat', 'ivf', 'hnsw'], default='flat', 
                      help='FAISS索引类型')
    parser.add_argument('--query', help='查询代码文件路径')
    parser.add_argument('--top-k', type=int, default=5, help='返回结果数量')
    parser.add_argument('--filter-type', help='按算法类型过滤结果')
    parser.add_argument('--interactive', action='store_true', help='交互式搜索模式')
    parser.add_argument('--save', action='store_true', help='保存索引')
    
    args = parser.parse_args()
    
    # 初始化搜索引擎
    search_engine = AlgorithmSemanticSearch(args.model)
    
    # 加载现有索引或创建新索引
    if args.index and args.metadata and os.path.exists(args.index) and os.path.exists(args.metadata):
        search_engine.load_index(args.index, args.metadata)
    elif args.repo:
        search_engine.create_index(args.index_type)
        search_engine.index_repository(args.repo)
        
        if args.save:
            index_path = args.index or 'algorithm_vectors.index'
            metadata_path = args.metadata or 'algorithm_metadata.pkl'
            search_engine.save_index(index_path, metadata_path)
    else:
        print("错误: 必须提供代码库路径或现有索引路径")
        return
    
    # 交互式搜索
    if args.interactive:
        interactive_search(search_engine)
    # 文件查询
    elif args.query:
        with open(args.query, 'r', encoding='utf-8') as f:
            query_code = f.read()
        
        results = search_engine.search(query_code, args.top_k, args.filter_type)
        display_results(results)


def interactive_search(search_engine):
    """交互式搜索界面"""
    import colorama
    from colorama import Fore, Style
    
    colorama.init()
    
    print(f"\n{Fore.CYAN}===== 算法语义代码搜索 ====={Style.RESET_ALL}")
    print(f"索引大小: {len(search_engine.metadata)} 个函数")
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
        
        # 查询参数
        top_k = 5
        try:
            top_k = int(input(f"{Fore.YELLOW}返回结果数量 [{top_k}]: {Style.RESET_ALL}") or top_k)
        except ValueError:
            print("输入无效，使用默认值")
        
        # 算法类型过滤
        filter_type = input(f"{Fore.YELLOW}按算法类型过滤 (留空表示自动): {Style.RESET_ALL}")
        filter_type = filter_type.strip() if filter_type.strip() else None
        
        # 执行搜索
        results = search_engine.search(query_code, top_k=top_k, filter_algo_type=filter_type)
        
        if not results:
            print(f"{Fore.RED}未找到相似代码{Style.RESET_ALL}")
            continue
        
        # 显示结果
        print(f"\n{Fore.GREEN}找到 {len(results)} 个相似函数:{Style.RESET_ALL}")
        
        for i, (metadata, distance, similarity) in enumerate(results):
            print(f"\n{Fore.CYAN}[{i+1}] {metadata['name']} (相似度: {similarity:.4f}){Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}文件: {metadata['path']}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}算法类型: {metadata['algorithm_type']}{Style.RESET_ALL}")
            
            # 显示代码片段
            code = metadata['code']
            MAX_DISPLAY_LINES = 20  # 限制显示的代码行数
            code_lines = code.split('\n')
            if len(code_lines) > MAX_DISPLAY_LINES:
                display_code = '\n'.join(code_lines[:MAX_DISPLAY_LINES]) + '\n...'
            else:
                display_code = code
                
            print(f"{display_code}\n")
            
            # 每次显示3个结果后暂停
            if (i + 1) % 3 == 0 and i + 1 < len(results):
                if input("继续显示更多结果? (y/n): ").lower() != 'y':
                    break
        
        # 询问是否继续搜索
        if input(f"\n{Fore.YELLOW}继续搜索? (y/n): {Style.RESET_ALL}").lower() != 'y':
            break
    
    print("搜索结束")


def display_results(results):
    """显示搜索结果"""
    print(f"\n找到 {len(results)} 个相似函数:")
    
    for i, (metadata, distance, similarity) in enumerate(results):
        print(f"\n[{i+1}] {metadata['name']} (相似度: {similarity:.4f})")
        print(f"文件: {metadata['path']}")
        print(f"算法类型: {metadata['algorithm_type']}")
        
        # 显示代码片段(截取前200字符)
        code = metadata['code']
        print(code[:200] + "..." if len(code) > 200 else code)


if __name__ == "__main__":
    main()