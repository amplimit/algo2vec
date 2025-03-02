# algo2vec

algo2vec是一个基于深度学习的算法与代码表示工具，能够将源代码转换为固定长度的向量表示，以支持多种代码理解和分析任务。受到word2vec和code2vec的启发，algo2vec专注于捕获算法和代码的语义特征。

## 项目特点

- 🔍 **语义代码理解**：通过深度学习捕获代码的语义特征，而不只是表面语法
- 🔄 **路径抽象**：使用AST路径来表示代码结构和关系
- ⚡ **注意力机制**：智能加权不同路径的重要性
- 🔮 **方法名称预测**：根据方法体推断合适的方法名
- 🔍 **代码搜索**：基于语义相似性而非简单文本匹配

## 安装

```bash
# 克隆仓库
git clone https://github.com/amplimit/algo2vec.git
cd algo2vec

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```python
from algo2vec import Algo2Vec

# 初始化
algo2vec = Algo2Vec()

# 分析代码
code = """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""

# 训练模型
algo2vec.train(code_samples, labels)

# 预测方法名
predictions = algo2vec.predict(code, top_k=3)
print(f"预测的方法名: {predictions}")

# 获取代码向量
code_vector = algo2vec.get_code_vector(code)

# 比较代码相似度
similar_codes = algo2vec.find_similar_codes(query_code, code_database, top_k=5)
for idx, similarity in similar_codes:
    print(f"相似度: {similarity:.2f}, 索引: {idx}")
```

## 项目结构

```
algo2vec/
├── algo2vec_core.py      # 核心功能实现
├── demo.py               # 演示应用
├── train.py              # 模型训练脚本
├── models/               # 预训练模型
├── data/                 # 示例数据
├── examples/             # 使用示例
├── tests/                # 测试
└── README.md             # 项目文档
```

## 功能详解

### 代码向量化

CodeInsight使用基于AST路径的表示方法，将代码转换为向量：

1. 将源代码解析为抽象语法树(AST)
2. 提取树中的路径上下文，每个上下文包括：
   - 起始终端值（变量名、类型等）
   - AST路径
   - 结束终端值
3. 将路径上下文映射到向量空间
4. 使用注意力机制聚合多个路径向量
5. 生成单一的代码向量表示

### 方法名称预测

通过学习代码结构与功能名称之间的关系，CodeInsight能够：

- 理解代码的功能意图
- 提出描述性的方法名称
- 检测命名不当的方法

### 代码相似性分析

CodeInsight可以计算代码片段之间的语义相似度，这对于以下任务非常有用：

- 代码克隆检测
- 代码搜索
- 代码推荐
- 代码重构建议

## 训练自己的模型

```python
from algo2vec import Algo2Vec

# 准备训练数据
code_samples = [
    """
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    """,
    # 更多代码示例...
]
labels = ["isPrime", ...]  # 对应的标签列表

# 初始化并训练
algo2vec = Algo2Vec(embedding_dim=128, hidden_dim=128)
algo2vec.train(
    code_samples=code_samples,
    labels=labels,
    epochs=20,
    batch_size=32,
    learning_rate=0.001,
    validation_split=0.1
)

# 保存模型
algo2vec.save_model("my_model.pt")

# 加载模型
algo2vec.load_model("my_model.pt")
```

## 示例应用

运行演示应用，体验algo2vec的功能：

```bash
python demo.py
```

演示应用展示了：
- 模型训练
- 代码分析
- 方法名称预测
- 代码语法和语义相似度计算
- 代码向量可视化
- 基于语义的代码搜索

## 未来计划

- [ ] 添加对Java、C++等更多编程语言的支持
- [ ] 改进路径提取算法和路径表示
- [ ] 集成Transformer架构以提高性能
- [ ] 在大规模代码库上训练预训练模型
- [ ] 支持代码生成和自动补全
- [ ] 开发VSCode插件和命令行工具
- [ ] 添加代码重构建议功能

## 贡献

欢迎贡献！请查看[贡献指南](CONTRIBUTING.md)了解如何参与项目。

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 特性对比

| 特性 | algo2vec | code2vec |
|------|---------|----------|
| 代码表示 | AST路径 | AST路径 |
| 注意力机制 | 软注意力 | 软注意力 |
| 深度学习框架 | PyTorch | TensorFlow |
| 多语言支持 | Python (可扩展) | Java |
| 代码相似度 | 语法+语义 | 不支持 |
| 代码搜索 | 支持 | 不支持 |
| 方法名预测 | 支持 | 支持 |
| 在线演示 | 本地演示 | Web演示 |
| GPU 加速 | 支持 | 支持 |

## 致谢

本项目受到[code2vec](https://github.com/tech-srl/code2vec)和[word2vec](https://code.google.com/archive/p/word2vec/)的启发，我们感谢这些项目的作者提供的研究和代码基础。