import os
import ast
import glob
import time
import random
from collections import Counter
import torch
from tqdm import tqdm

# 导入Algo2Vec类
from core import Algo2Vec

# 配置参数
REPO_PATH = "./dataset/code1"
OUTPUT_MODEL_PATH = "algo2vec_repository_model.pt"
MIN_FUNCTION_LINES = 3  # 忽略非常短的函数
MAX_FUNCTION_LINES = 100  # 忽略非常长的函数
MAX_SAMPLES_PER_LABEL = 1000  # 每个标签最多使用的样本数
MIN_SAMPLES_PER_LABEL = 5  # 每个标签最少需要的样本数
VALIDATION_SPLIT = 0.1
MAX_TOTAL_SAMPLES = 50000  # 限制总样本数，避免内存问题
RANDOM_SEED = 42

# 设置随机种子以保证可重复性
random.seed(RANDOM_SEED)

class FunctionVisitor(ast.NodeVisitor):
    """访问AST以提取函数和方法定义"""
    
    def __init__(self):
        self.functions = []
        self.current_class = None
    
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        # 访问类的内容
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        # 获取函数的源代码
        function_lines = []
        for i in range(node.lineno - 1, node.end_lineno):
            function_lines.append(self.source_lines[i])
        source_code = '\n'.join(function_lines)
        
        # 确保函数有足够的行数
        if len(function_lines) >= MIN_FUNCTION_LINES and len(function_lines) <= MAX_FUNCTION_LINES:
            if self.current_class:
                # 这是一个方法，添加类名前缀
                function_name = f"{self.current_class}.{node.name}"
            else:
                # 这是一个普通函数
                function_name = node.name
            
            # 过滤掉私有函数和测试函数（可选）
            if not node.name.startswith('_') and not node.name.startswith('test_'):
                self.functions.append((function_name, source_code))
        
        # 继续访问函数内部（可能有嵌套函数）
        self.generic_visit(node)
    
    def extract_functions(self, source_code, filename):
        """从源代码中提取所有函数和方法"""
        self.functions = []
        self.source_lines = source_code.split('\n')
        
        try:
            tree = ast.parse(source_code)
            self.visit(tree)
            return self.functions
        except SyntaxError as e:
            print(f"语法错误: {filename} - {e}")
            return []
        except Exception as e:
            print(f"解析错误: {filename} - {e}")
            return []

def find_python_files(repo_path):
    """查找所有Python文件"""
    print(f"在 {repo_path} 中查找Python文件...")
    python_files = []
    
    # 使用glob递归查找所有.py文件
    for py_file in glob.glob(os.path.join(repo_path, '**', '*.py'), recursive=True):
        # 过滤掉测试文件和迁移文件（可选）
        filename = os.path.basename(py_file)
        if not filename.startswith('test_') and 'migration' not in py_file.lower():
            python_files.append(py_file)
    
    print(f"找到 {len(python_files)} 个Python文件")
    return python_files

def extract_functions_from_files(python_files):
    """从Python文件中提取函数"""
    all_functions = []
    visitor = FunctionVisitor()
    
    print("正在从文件中提取函数...")
    for py_file in tqdm(python_files):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            functions = visitor.extract_functions(source_code, py_file)
            all_functions.extend(functions)
        except UnicodeDecodeError:
            try:
                # 尝试其他编码
                with open(py_file, 'r', encoding='latin-1') as f:
                    source_code = f.read()
                functions = visitor.extract_functions(source_code, py_file)
                all_functions.extend(functions)
            except Exception as e:
                print(f"读取错误: {py_file} - {e}")
        except Exception as e:
            print(f"处理错误: {py_file} - {e}")
    
    print(f"共提取了 {len(all_functions)} 个函数和方法")
    return all_functions

def process_function_names(function_names):
    """处理函数名称，提取有意义的标签"""
    processed_labels = []
    
    for name in function_names:
        # 如果是方法，只保留方法名部分
        if '.' in name:
            name = name.split('.')[-1]
        
        # 处理常见命名风格（驼峰命名、下划线分隔）
        if '_' in name:
            # 处理snake_case
            parts = name.split('_')
            main_verb = parts[0].lower()
        else:
            # 处理camelCase或PascalCase
            # 找到第一个大写字母之前的部分作为主要动词
            main_verb = ''
            for char in name:
                if char.isupper() and main_verb:
                    break
                main_verb += char.lower()
        
        # 如果主要动词太短，使用整个函数名
        if len(main_verb) < 2:
            main_verb = name.lower()
        
        # 过滤掉无意义的标签
        if main_verb in ['get', 'set', 'is', 'do', 'a', 'an', 'the']:
            main_verb = name.lower()
        
        processed_labels.append(main_verb)
    
    return processed_labels

def prepare_training_data(all_functions):
    """准备训练数据"""
    function_names = [name for name, _ in all_functions]
    function_codes = [code for _, code in all_functions]
    
    # 处理函数名称，提取标签
    labels = process_function_names(function_names)
    
    # 分析标签分布
    label_counts = Counter(labels)
    print("\n标签分布（前20个）:")
    for label, count in label_counts.most_common(20):
        print(f"  {label}: {count}")
    
    # 过滤掉样本数太少的标签
    valid_labels = {label for label, count in label_counts.items() if count >= MIN_SAMPLES_PER_LABEL}
    print(f"\n至少有{MIN_SAMPLES_PER_LABEL}个样本的标签数量: {len(valid_labels)}")
    
    # 准备训练数据，保持每个标签的平衡
    balanced_codes = []
    balanced_labels = []
    
    label_sample_counts = {label: 0 for label in valid_labels}
    
    # 打乱数据
    combined = list(zip(function_codes, labels))
    random.shuffle(combined)
    function_codes, labels = zip(*combined)
    
    # 选择平衡的样本
    for code, label in zip(function_codes, labels):
        if label in valid_labels and label_sample_counts[label] < MAX_SAMPLES_PER_LABEL:
            balanced_codes.append(code)
            balanced_labels.append(label)
            label_sample_counts[label] += 1
        
        # 限制总样本数
        if len(balanced_codes) >= MAX_TOTAL_SAMPLES:
            break
    
    print(f"\n准备了 {len(balanced_codes)} 个平衡样本，共 {len(set(balanced_labels))} 个不同标签")
    
    return balanced_codes, balanced_labels

def train_model(codes, labels):
    """训练algo2vec模型"""
    print("\n开始训练模型...")
    
    # 创建Algo2Vec实例
    algo2vec = Algo2Vec(embedding_dim=128, hidden_dim=256)
    
    # 训练模型
    start_time = time.time()
    algo2vec.train(
        code_samples=codes,
        labels=labels,
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        validation_split=VALIDATION_SPLIT
    )
    
    training_time = time.time() - start_time
    print(f"训练完成，耗时 {training_time:.2f} 秒")
    
    # 保存模型
    algo2vec.save_model(OUTPUT_MODEL_PATH)
    print(f"模型已保存到: {OUTPUT_MODEL_PATH}")
    
    return algo2vec

def main():
    """主函数"""
    print("开始处理代码库...")
    
    # 查找Python文件
    python_files = find_python_files(REPO_PATH)
    
    # 提取函数
    all_functions = extract_functions_from_files(python_files)
    
    # 准备训练数据
    codes, labels = prepare_training_data(all_functions)
    
    # 检查是否有足够的训练数据
    if len(codes) < 100:
        print("警告: 训练样本太少，可能导致训练效果不佳")
        if input("是否继续? (y/n): ").lower() != 'y':
            return
    
    # 训练模型
    algo2vec = train_model(codes, labels)
    
    print("\n模型训练完成!")
    print(f"你可以使用以下代码加载模型:")
    print(f"```python")
    print(f"from core import Algo2Vec")
    print(f"algo2vec = Algo2Vec()")
    print(f"algo2vec.load_model('{OUTPUT_MODEL_PATH}')")
    print(f"```")

if __name__ == "__main__":
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 没有检测到GPU，训练可能较慢")
    
    main()