import sys
import os
from typing import List, Dict, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import random

# 导入Algo2Vec核心模块
sys.path.append(".")
from core import Algo2Vec, PathExtractor, CodeVectorizer


class Algo2VecDemo:
    """Algo2Vec演示应用"""
    
    def __init__(self, model_path: str = None):
        self.algo2vec = Algo2Vec(embedding_dim=128, hidden_dim=128)
        self.extractor = PathExtractor()
        if model_path and os.path.exists(model_path):
            self.algo2vec.load_model(model_path)
            self.model_loaded = True
        else:
            print("没有找到预训练模型，将使用随机初始化模型进行演示。")
            self.model_loaded = False
        
    def train_demo_model(self, save_path: str = "algo2vec_demo_model.pt"):
        """训练一个演示模型"""
        # 准备训练数据
        code_samples = []
        labels = []
        
        # 素数检测
        for i in range(10):
            code_samples.append(f"""
def isPrime{i}(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
            """)
            labels.append("isPrime")
        
        # 数组反转
        for i in range(10):
            code_samples.append(f"""
def reverseArray{i}(arr):
    result = [0] * len(arr)
    for i in range(len(arr)):
        result[len(arr) - 1 - i] = arr[i]
    return result
            """)
            labels.append("reverseArray")
        
        # 数组排序
        for i in range(10):
            code_samples.append(f"""
def sortArray{i}(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
            """)
            labels.append("sortArray")
        
        # 求和函数
        for i in range(10):
            code_samples.append(f"""
def sumArray{i}(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
            """)
            labels.append("sumArray")
        
        # 查找最大值
        for i in range(10):
            code_samples.append(f"""
def findMax{i}(arr):
    if not arr:
        return None
    max_val = arr[0]
    for val in arr:
        if val > max_val:
            max_val = val
    return max_val
            """)
            labels.append("findMax")
        
        print(f"准备了{len(code_samples)}个训练样本，开始训练...")
        
        # 训练模型
        self.algo2vec.train(
            code_samples=code_samples,
            labels=labels,
            epochs=10,
            batch_size=8,
            learning_rate=0.001,
            validation_split=0.2
        )
        
        # 保存模型
        self.algo2vec.save_model(save_path)
        self.model_loaded = True
        
        print(f"演示模型已训练并保存到: {save_path}")
    
    def predict_method_name(self, code: str, top_k: int = 3) -> List[str]:
        """预测方法名称"""
        if not self.model_loaded:
            # 如果模型未加载，使用启发式规则预测
            if "prime" in code.lower() or ("n % i" in code and "return False" in code):
                return ["isPrime", "checkPrimality", "validateNumber"]
            elif "reverse" in code.lower() or "[len(arr) - i - 1]" in code or "[len(arr) - 1 - i]" in code:
                return ["reverseArray", "reverse", "flipArray"]
            elif "sort" in code.lower() or ("for i in range" in code and "for j in range" in code and "arr[j] > arr[j + 1]" in code):
                return ["sortArray", "bubbleSort", "sort"]
            elif "total = 0" in code and "total +=" in code:
                return ["sumArray", "calculateSum", "sum"]
            elif "max_val" in code or "max =" in code:
                return ["findMax", "getMaximum", "maximum"]
            else:
                return ["processData", "compute", "execute"]
        else:
            # 使用训练好的模型预测
            try:
                predictions = self.algo2vec.predict(code, top_k=top_k)
                if isinstance(predictions, list):
                    return [name for name, _ in predictions]
                else:
                    return [predictions]
            except Exception as e:
                print(f"预测出错: {e}")
                return ["unknown"]
    
    def analyze_code(self, code: str) -> Dict:
        """分析代码并返回结果"""
        # 提取路径
        paths = self.extractor.extract_paths(code)
        
        # 预测方法名
        predicted_names = self.predict_method_name(code)
        
        # 计算代码复杂度
        complexity = self._calculate_complexity(code)
        
        # 获取代码向量（如果模型已加载）
        if self.model_loaded:
            try:
                code_vector = self.algo2vec.get_code_vector(code)
                vec_norm = np.linalg.norm(code_vector)
                vector_stats = {
                    "norm": float(vec_norm),
                    "mean": float(np.mean(code_vector)),
                    "std": float(np.std(code_vector))
                }
            except Exception as e:
                print(f"获取代码向量出错: {e}")
                vector_stats = None
        else:
            vector_stats = None
        
        # 返回分析结果
        return {
            "path_count": len(paths),
            "predicted_names": predicted_names,
            "complexity": complexity,
            "vector_stats": vector_stats,
            "code_summary": self._generate_summary(code, predicted_names[0])
        }
    
    def _calculate_complexity(self, code: str) -> Dict:
        """计算代码复杂度指标"""
        lines = [line for line in code.split("\n") if line.strip()]
        
        # 计算循环嵌套深度
        max_loop_depth = 0
        current_loop_depth = 0
        for line in lines:
            if "for " in line or "while " in line:
                current_loop_depth += 1
                max_loop_depth = max(max_loop_depth, current_loop_depth)
            elif line.strip().startswith("}") or (line.strip() and line.count(" ") < current_loop_depth * 4):
                # 简单的启发式检测循环结束（不完美，但演示够用）
                current_loop_depth = max(0, current_loop_depth - 1)
        
        # 计算条件语句数量
        if_count = sum(1 for line in lines if "if " in line)
        
        # 其他指标
        return {
            "lines": len(lines),
            "max_loop_depth": max_loop_depth,
            "if_count": if_count,
            "cyclomatic_complexity": if_count + max_loop_depth + 1  # 简化版的圈复杂度
        }
    
    def _generate_summary(self, code: str, predicted_name: str) -> str:
        """生成代码摘要"""
        complexity = self._calculate_complexity(code)
        
        # 根据复杂度和预测的方法名生成摘要
        if complexity["max_loop_depth"] > 1:
            loop_desc = "嵌套循环"
        elif complexity["max_loop_depth"] == 1:
            loop_desc = "一个循环"
        else:
            loop_desc = "无循环"
            
        if complexity["if_count"] > 2:
            cond_desc = "多个条件分支"
        elif complexity["if_count"] > 0:
            cond_desc = f"{complexity['if_count']}个条件分支"
        else:
            cond_desc = "无条件分支"
            
        if complexity["lines"] > 15:
            size_desc = "较长"
        elif complexity["lines"] > 8:
            size_desc = "中等长度"
        else:
            size_desc = "简短"
            
        # 生成摘要
        return f"这段代码是一个{size_desc}的{predicted_name}方法，包含{loop_desc}和{cond_desc}。"
    
    def compare_codes(self, code1: str, code2: str) -> Dict:
        """比较两段代码的相似度"""
        # 提取路径用于语法相似度
        paths1 = set(str(p) for p in self.extractor.extract_paths(code1))
        paths2 = set(str(p) for p in self.extractor.extract_paths(code2))
        
        # 计算Jaccard语法相似度
        intersection = len(paths1.intersection(paths2))
        union = len(paths1.union(paths2))
        syntax_similarity = intersection / union if union > 0 else 0
        
        # 计算语义相似度（如果模型已加载）
        if self.model_loaded:
            try:
                vec1 = self.algo2vec.get_code_vector(code1)
                vec2 = self.algo2vec.get_code_vector(code2)
                
                # 计算余弦相似度
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    semantic_similarity = 0.0
                else:
                    semantic_similarity = dot_product / (norm1 * norm2)
            except Exception as e:
                print(f"计算语义相似度出错: {e}")
                semantic_similarity = None
        else:
            semantic_similarity = None
        
        return {
            "syntax_similarity": syntax_similarity,
            "semantic_similarity": semantic_similarity,
            "common_paths": intersection,
            "total_unique_paths": union
        }
    
    def visualize_code_vectors(self, code_samples: List[str], labels: List[str]):
        """可视化多个代码片段的向量"""
        if not self.model_loaded:
            # 使用随机向量用于演示
            print("使用随机向量进行可视化（模型未加载）")
            vectors = np.random.rand(len(code_samples), 128)
        else:
            # 使用模型获取代码向量
            try:
                print("使用模型生成代码向量进行可视化")
                vectors = np.array([self.algo2vec.get_code_vector(code) for code in code_samples])
            except Exception as e:
                print(f"获取代码向量出错: {e}")
                # 失败时回退到随机向量
                vectors = np.random.rand(len(code_samples), 128)
        
        # 使用t-SNE降维到2D进行可视化
        # 设置perplexity为样本数量-1，确保它小于样本数量
        perplexity = min(3, len(code_samples) - 1)  # 如果样本数量小于等于4，使用最小值3
        if len(code_samples) <= 3:
            # 样本太少，无法使用t-SNE，直接使用PCA
            from sklearn.decomposition import PCA
            print(f"样本数量过少({len(code_samples)})，使用PCA而非t-SNE进行降维")
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            print(f"使用t-SNE进行降维，perplexity设置为{perplexity}")
            vectors_2d = tsne.fit_transform(vectors)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        
        # 为不同类别的代码使用不同颜色
        unique_labels = list(set(labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, (vec, label) in enumerate(zip(vectors_2d, labels)):
            color_idx = unique_labels.index(label)
            plt.scatter(vec[0], vec[1], color=colors[color_idx], 
                       label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.text(vec[0], vec[1], f"{i}", fontsize=9)
        
        plt.legend()
        plt.title("代码向量可视化")
        plt.savefig("code_vectors.png")
        plt.close()
        
        return "code_vectors.png"
    
    def search_similar_code(self, query_code: str, code_database: List[str], top_k: int = 3) -> List[Tuple[int, float, str]]:
        """搜索与查询代码最相似的代码片段"""
        if not self.model_loaded:
            print("模型未加载，使用语法相似度进行搜索")
            # 使用语法相似度
            query_paths = set(str(p) for p in self.extractor.extract_paths(query_code))
            results = []
            
            for i, code in enumerate(code_database):
                code_paths = set(str(p) for p in self.extractor.extract_paths(code))
                
                # 计算Jaccard相似度
                intersection = len(query_paths.intersection(code_paths))
                union = len(query_paths.union(code_paths))
                similarity = intersection / union if union > 0 else 0
                
                results.append((i, similarity, self._get_code_snippet(code)))
            
            # 按相似度排序
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        else:
            # 使用语义相似度
            try:
                similar_indices = self.algo2vec.find_similar_codes(query_code, code_database, top_k)
                
                # 格式化结果
                results = [(idx, sim, self._get_code_snippet(code_database[idx])) 
                          for idx, sim in similar_indices]
                return results
            except Exception as e:
                print(f"搜索相似代码出错: {e}")
                return []
    
    def _get_code_snippet(self, code: str, max_length: int = 100) -> str:
        """获取代码的简短摘要用于展示"""
        code = code.strip()
        if len(code) <= max_length:
            return code
        
        # 提取函数签名和前几行
        lines = code.split('\n')
        signature = lines[0] if lines else ""
        
        snippet = signature
        remaining_len = max_length - len(snippet)
        
        if remaining_len > 20:  # 确保至少有足够空间展示一些内容
            snippet += "\n" + "\n".join(lines[1:3])
            snippet += "\n..."
        
        return snippet


if __name__ == "__main__":
    # 示例代码
    example_codes = [
        """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
        """,
        
        """
def reverse_array(arr):
    result = [0] * len(arr)
    for i in range(len(arr)):
        result[len(arr) - i - 1] = arr[i]
    return result
        """,
        
        """
def sum_array(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
        """,
        
        """
def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for val in arr:
        if val > max_val:
            max_val = val
    return max_val
        """,
        
        """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
        """
    ]
    
    # 定义方法名称
    method_names = ["isPrime", "reverseArray", "sumArray", "findMax", "bubbleSort"]
    
    # 创建代码数据库（稍微变化的代码）
    code_database = []
    for i in range(5):
        for j, code in enumerate(example_codes):
            # 添加一些随机变化
            modified_code = code.replace("arr", f"array{random.randint(0,9)}")
            modified_code = modified_code.replace("total", f"sum{random.randint(0,9)}")
            modified_code = modified_code.replace("val", f"value{random.randint(0,9)}")
            code_database.append(modified_code)
    
    # 检查是否有预训练模型
    model_path = "algo2vec_demo_model.pt"
    
    # 初始化演示应用
    demo = Algo2VecDemo(model_path)
    
    print("=== Algo2Vec 演示 ===")
    
    # 训练模型（如果没有找到预训练模型）
    if not demo.model_loaded:
        print("\n1. 训练演示模型")
        demo.train_demo_model(model_path)
    
    # 分析代码
    print("\n2. 分析代码")
    for i, code in enumerate(example_codes):
        print(f"\n代码片段 {i+1}:")
        print(code.strip())
        
        result = demo.analyze_code(code)
        print(f"\n分析结果:")
        print(f"  路径数量: {result['path_count']}")
        print(f"  预测的方法名: {', '.join(result['predicted_names'])}")
        print(f"  代码复杂度: {result['complexity']}")
        
        if result['vector_stats']:
            print(f"  向量范数: {result['vector_stats']['norm']:.2f}")
            print(f"  向量均值: {result['vector_stats']['mean']:.2f}")
        
        print(f"  代码摘要: {result['code_summary']}")
    
    # 比较代码
    print("\n3. 比较代码相似度")
    similarity = demo.compare_codes(example_codes[0], example_codes[1])
    print(f"素数检测 vs 数组反转:")
    print(f"  语法相似度: {similarity['syntax_similarity']:.2f}")
    if similarity['semantic_similarity'] is not None:
        print(f"  语义相似度: {similarity['semantic_similarity']:.2f}")
    
    similarity = demo.compare_codes(example_codes[1], example_codes[2])
    print(f"数组反转 vs 数组求和:")
    print(f"  语法相似度: {similarity['syntax_similarity']:.2f}")
    if similarity['semantic_similarity'] is not None:
        print(f"  语义相似度: {similarity['semantic_similarity']:.2f}")
    
    similarity = demo.compare_codes(example_codes[3], example_codes[4])
    print(f"查找最大值 vs 冒泡排序:")
    print(f"  语法相似度: {similarity['syntax_similarity']:.2f}")
    if similarity['semantic_similarity'] is not None:
        print(f"  语义相似度: {similarity['semantic_similarity']:.2f}")
    
    # 可视化
    print("\n4. 可视化代码向量")
    image_path = demo.visualize_code_vectors(example_codes, method_names)
    print(f"可视化结果已保存到: {image_path}")
    
    # 代码搜索
    print("\n5. 代码搜索")
    query_code = """
def find_minimum(numbers):
    if not numbers:
        return None
    min_value = numbers[0]
    for number in numbers:
        if number < min_value:
            min_value = number
    return min_value
    """
    
    print("查询代码:")
    print(query_code.strip())
    
    print("\n搜索结果:")
    results = demo.search_similar_code(query_code, code_database, top_k=3)
    for i, (idx, similarity, snippet) in enumerate(results):
        print(f"\n结果 {i+1} (相似度: {similarity:.2f}):")
        print(snippet)
    
    print("\n演示完成！")