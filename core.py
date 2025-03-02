import os
import ast
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from typing import List, Dict, Tuple, Set, Optional, Union
from tqdm import tqdm


class PathExtractor:
    """从Python AST中提取路径的组件"""
    
    def __init__(self, max_path_length: int = 8, max_path_width: int = 2):
        self.max_path_length = max_path_length
        self.max_path_width = max_path_width
        
    def extract_paths(self, code: str) -> List[Tuple]:
        """从代码字符串中提取路径上下文"""
        try:
            tree = ast.parse(code)
            paths = []
            self._extract_paths_from_node(tree, [], paths)
            return paths
        except SyntaxError:
            return []
    
    def _extract_paths_from_node(self, node, current_path, paths):
        """递归遍历AST并提取路径"""
        # 终止条件: 到达叶子节点或路径长度超过最大值
        if len(current_path) > self.max_path_length:
            return
            
        # 处理当前节点
        if isinstance(node, ast.Name):
            # 对于变量名节点，我们提取它的值
            if current_path:  # 如果路径不为空，我们有一个完整路径
                paths.append((node.id, tuple(current_path), None))
        
        elif isinstance(node, ast.Constant):
            # 对于常量节点，我们提取它的值
            if current_path:
                paths.append((str(node.value), tuple(current_path), None))
                
        # 递归处理子节点
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                # 如果字段是列表，递归处理每个项
                for item in value:
                    if isinstance(item, ast.AST):
                        new_path = current_path + [(field, type(node).__name__, "down")]
                        self._extract_paths_from_node(item, new_path, paths)
            elif isinstance(value, ast.AST):
                # 如果字段是单个AST节点，递归处理它
                new_path = current_path + [(field, type(node).__name__, "down")]
                self._extract_paths_from_node(value, new_path, paths)


class CodeVectorizer:
    """将代码转换为向量表示的组件"""
    
    def __init__(self, embedding_dim: int = 128, vocab_size: int = 10000):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.path_extractor = PathExtractor()
        self.token_to_index = {}
        self.path_to_index = {}
        self.next_token_index = 1  # 0 保留给未知标记
        self.next_path_index = 1
        
    def build_vocab(self, code_samples: List[str]):
        """从代码样本中构建词汇表"""
        for code in code_samples:
            paths = self.path_extractor.extract_paths(code)
            for start_token, path, end_token in paths:
                if start_token not in self.token_to_index:
                    self.token_to_index[start_token] = self.next_token_index
                    self.next_token_index += 1
                
                if end_token and end_token not in self.token_to_index:
                    self.token_to_index[end_token] = self.next_token_index
                    self.next_token_index += 1
                
                path_str = str(path)
                if path_str not in self.path_to_index:
                    self.path_to_index[path_str] = self.next_path_index
                    self.next_path_index += 1
    
    def vectorize(self, code: str) -> List[Tuple[int, int, int]]:
        """将代码转换为路径上下文向量"""
        paths = self.path_extractor.extract_paths(code)
        result = []
        
        for start_token, path, end_token in paths:
            start_idx = self.token_to_index.get(start_token, 0)
            path_str = str(path)
            path_idx = self.path_to_index.get(path_str, 0)
            end_idx = self.token_to_index.get(end_token, 0) if end_token else 0
            
            result.append((start_idx, path_idx, end_idx))
            
        return result


class PathAttentionNetwork(nn.Module):
    """使用注意力机制的路径聚合网络"""
    
    def __init__(self, token_vocab_size: int, path_vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128):
        super(PathAttentionNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 嵌入层
        self.token_embedding = nn.Embedding(token_vocab_size, embedding_dim)
        self.path_embedding = nn.Embedding(path_vocab_size, embedding_dim)
        
        # 全连接层
        self.fc = nn.Linear(3 * embedding_dim, hidden_dim)
        
        # 注意力向量
        self.attention = nn.Parameter(torch.randn(hidden_dim))
        
        # 输出层
        self.output_fc = nn.Linear(hidden_dim, hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.path_embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.output_fc.weight)
        nn.init.zeros_(self.output_fc.bias)
        # 对于一维向量，使用普通的正态分布初始化
        nn.init.normal_(self.attention.data, mean=0, std=0.1)
    
    def forward(self, path_contexts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        # 处理空的路径上下文列表
        if not path_contexts:
            return torch.zeros(self.hidden_dim, device=self.device)
        
        # 解包路径上下文
        start_tokens, paths, end_tokens = zip(*path_contexts)
        
        # 将列表转换为张量
        start_tokens = torch.stack(start_tokens).to(self.device)
        paths = torch.stack(paths).to(self.device)
        end_tokens = torch.stack(end_tokens).to(self.device)
        
        # 嵌入路径上下文的各个部分
        start_embedded = self.token_embedding(start_tokens)
        path_embedded = self.path_embedding(paths)
        end_embedded = self.token_embedding(end_tokens)
        
        # 连接嵌入
        context_embedded = torch.cat([start_embedded, path_embedded, end_embedded], dim=1)
        
        # 通过全连接层
        context_encoded = torch.tanh(self.fc(context_embedded))
        
        # 计算注意力分数 (context x attention)
        attention_scores = torch.matmul(context_encoded, self.attention)
        
        # 应用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # 加权聚合形成代码向量
        code_vector = torch.sum(attention_weights.unsqueeze(1) * context_encoded, dim=0)
        
        # 应用输出层
        output = self.output_fc(code_vector)
        
        return output


class CodeDataset(Dataset):
    """代码数据集，用于批处理训练"""
    
    def __init__(self, code_samples: List[str], labels: List[int], vectorizer):
        self.code_samples = code_samples
        self.labels = labels
        self.vectorizer = vectorizer
        
    def __len__(self):
        return len(self.code_samples)
    
    def __getitem__(self, idx):
        code = self.code_samples[idx]
        label = self.labels[idx]
        
        # 向量化代码
        vectorized_code = self.vectorizer.vectorize(code)
        
        # 将路径上下文转换为张量
        contexts = []
        for start_idx, path_idx, end_idx in vectorized_code:
            contexts.append((
                torch.tensor(start_idx, dtype=torch.long),
                torch.tensor(path_idx, dtype=torch.long),
                torch.tensor(end_idx, dtype=torch.long)
            ))
            
        return contexts, torch.tensor(label, dtype=torch.long)
    
    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        contexts_batch = []
        labels = []
        
        for contexts, label in batch:
            contexts_batch.append(contexts)
            labels.append(label)
            
        return contexts_batch, torch.stack(labels)


class Algo2Vec:
    """Algo2Vec主类，整合所有组件"""
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.code_vectorizer = CodeVectorizer(embedding_dim)
        self.model = None
        self.label_to_index = {}
        self.index_to_label = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self, code_samples: List[str], labels: List[str], epochs: int = 10, batch_size: int = 32, 
              learning_rate: float = 0.001, validation_split: float = 0.1):
        """训练模型"""
        print(f"开始训练Algo2Vec模型... 使用设备: {self.device}")
        
        # 构建词汇表
        print("构建词汇表...")
        self.code_vectorizer.build_vocab(code_samples)
        
        # 准备标签词汇表
        for label in labels:
            if label not in self.label_to_index:
                idx = len(self.label_to_index)
                self.label_to_index[label] = idx
                self.index_to_label[idx] = label
        
        # 将标签转换为索引
        label_indices = [self.label_to_index[label] for label in labels]
        
        # 打印标签分布，以检查数据集是否平衡
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print("标签分布:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
        
        # 分割训练集和验证集
        if validation_split > 0:
            # 保持每个类的比例
            train_codes = []
            train_labels = []
            val_codes = []
            val_labels = []
            
            # 对每个标签分别划分
            for label in set(labels):
                # 找出所有带有此标签的示例
                indices = [i for i, l in enumerate(labels) if l == label]
                # 打乱顺序
                np.random.shuffle(indices)
                # 计算验证集大小
                val_size = max(1, int(len(indices) * validation_split))
                # 划分
                val_indices = indices[:val_size]
                train_indices = indices[val_size:]
                
                # 添加到各自的列表中
                train_codes.extend([code_samples[i] for i in train_indices])
                train_labels.extend([label_indices[i] for i in train_indices])
                val_codes.extend([code_samples[i] for i in val_indices])
                val_labels.extend([label_indices[i] for i in val_indices])
            
            print(f"训练集大小: {len(train_codes)}, 验证集大小: {len(val_codes)}")
        else:
            train_codes = code_samples
            train_labels = label_indices
            val_codes = []
            val_labels = []
        
        # 初始化模型
        token_vocab_size = self.code_vectorizer.next_token_index
        path_vocab_size = self.code_vectorizer.next_path_index
        label_vocab_size = len(self.label_to_index)
        
        print(f"词汇表大小 - 标记: {token_vocab_size}, 路径: {path_vocab_size}, 标签: {label_vocab_size}")
        
        # 创建模型
        attention_network = PathAttentionNetwork(
            token_vocab_size, path_vocab_size, self.embedding_dim, self.hidden_dim
        ).to(self.device)
        
        classifier = nn.Linear(self.hidden_dim, label_vocab_size).to(self.device)
        
        # 创建训练数据集和数据加载器
        train_dataset = CodeDataset(train_codes, train_labels, self.code_vectorizer)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(batch_size, len(train_codes)), # 确保batch_size不大于训练样本数量
            shuffle=True, 
            collate_fn=CodeDataset.collate_fn
        )
        
        # 创建验证数据集和数据加载器（如果有）
        if val_codes:
            val_dataset = CodeDataset(val_codes, val_labels, self.code_vectorizer)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=min(batch_size, len(val_codes)), # 确保batch_size不大于验证样本数量
                shuffle=False, 
                collate_fn=CodeDataset.collate_fn
            )
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(attention_network.parameters()) + list(classifier.parameters()), 
            lr=learning_rate
        )
        
        # 使用学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=2, factor=0.5, verbose=True
        )
        
        # 训练循环
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        print(f"开始训练，共{epochs}个epochs...")
        for epoch in range(epochs):
            # 训练阶段
            attention_network.train()
            classifier.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_contexts, batch_labels in progress_bar:
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向传播
                batch_code_vectors = []
                for contexts in batch_contexts:
                    if not contexts:  # 处理空路径的情况
                        # 如果没有路径，使用零向量
                        code_vector = torch.zeros(self.hidden_dim, device=self.device)
                    else:
                        code_vector = attention_network(contexts)
                    batch_code_vectors.append(code_vector)
                
                # 堆叠批处理向量
                batch_code_vectors = torch.stack(batch_code_vectors)
                
                # 分类
                logits = classifier(batch_code_vectors)
                
                # 计算损失
                loss = criterion(logits, batch_labels.to(self.device))
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                optimizer.step()
                
                # 累积损失
                train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            
            # 验证阶段（如果有）
            if val_codes:
                attention_network.eval()
                classifier.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_contexts, batch_labels in val_loader:
                        # 前向传播
                        batch_code_vectors = []
                        for contexts in batch_contexts:
                            if not contexts:
                                code_vector = torch.zeros(self.hidden_dim, device=self.device)
                            else:
                                code_vector = attention_network(contexts)
                            batch_code_vectors.append(code_vector)
                        
                        # 堆叠批处理向量
                        batch_code_vectors = torch.stack(batch_code_vectors)
                        
                        # 分类
                        logits = classifier(batch_code_vectors)
                        
                        # 计算损失
                        loss = criterion(logits, batch_labels.to(self.device))
                        val_loss += loss.item()
                        
                        # 计算准确率
                        _, predicted = torch.max(logits, 1)
                        total += batch_labels.size(0)
                        correct += (predicted == batch_labels.to(self.device)).sum().item()
                
                # 计算平均验证损失和准确率
                val_loss /= len(val_loader)
                val_accuracy = 100 * correct / total
                
                # 更新学习率
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Accuracy: {val_accuracy:.2f}%")
                
                # 打印混淆矩阵
                if correct > 0 and epoch % 5 == 0:  # 每5个epoch打印一次
                    print("\n预测标签分布:")
                    pred_counts = {}
                    with torch.no_grad():
                        for batch_contexts, batch_labels in val_loader:
                            # 前向传播过程同上...
                            batch_code_vectors = []
                            for contexts in batch_contexts:
                                if not contexts:
                                    code_vector = torch.zeros(self.hidden_dim, device=self.device)
                                else:
                                    code_vector = attention_network(contexts)
                                batch_code_vectors.append(code_vector)
                            
                            batch_code_vectors = torch.stack(batch_code_vectors)
                            logits = classifier(batch_code_vectors)
                            _, predicted = torch.max(logits, 1)
                            
                            for pred in predicted:
                                label = self.index_to_label[pred.item()]
                                pred_counts[label] = pred_counts.get(label, 0) + 1
                    
                    for label, count in pred_counts.items():
                        print(f"  {label}: {count}")
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    self.model = {
                        "attention_network": attention_network,
                        "classifier": classifier
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
                self.model = {
                    "attention_network": attention_network,
                    "classifier": classifier
                }
        
        # 如果没有验证集或者没有触发早停，保存最后的模型
        if not val_codes or patience_counter < patience:
            self.model = {
                "attention_network": attention_network,
                "classifier": classifier
            }
        
        print("训练完成！")
        
    def predict(self, code: str, top_k: int = 1) -> Union[str, List[Tuple[str, float]]]:
        """预测代码的标签"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 将代码转换为向量表示
        vectorized = self.code_vectorizer.vectorize(code)
        
        # 准备数据
        contexts = []
        for start_idx, path_idx, end_idx in vectorized:
            contexts.append((
                torch.tensor(start_idx, dtype=torch.long, device=self.device),
                torch.tensor(path_idx, dtype=torch.long, device=self.device),
                torch.tensor(end_idx, dtype=torch.long, device=self.device)
            ))
        
        # 使用模型进行预测
        self.model["attention_network"].eval()
        self.model["classifier"].eval()
        
        with torch.no_grad():
            # 如果没有路径上下文，使用零向量
            if not contexts:
                code_vector = torch.zeros(self.hidden_dim, device=self.device)
            else:
                code_vector = self.model["attention_network"](contexts)
            
            # 分类
            logits = self.model["classifier"](code_vector)
            probabilities = F.softmax(logits, dim=0)
            
            # 获取top-k预测
            if top_k == 1:
                pred_idx = torch.argmax(probabilities).item()
                return self.index_to_label[pred_idx]
            else:
                top_k_values, top_k_indices = torch.topk(probabilities, min(top_k, len(self.index_to_label)))
                return [(self.index_to_label[idx.item()], prob.item()) for idx, prob in zip(top_k_indices, top_k_values)]
    
    def get_code_vector(self, code: str) -> np.ndarray:
        """获取代码的向量表示"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 将代码转换为向量表示
        vectorized = self.code_vectorizer.vectorize(code)
        
        # 准备数据
        contexts = []
        for start_idx, path_idx, end_idx in vectorized:
            contexts.append((
                torch.tensor(start_idx, dtype=torch.long, device=self.device),
                torch.tensor(path_idx, dtype=torch.long, device=self.device),
                torch.tensor(end_idx, dtype=torch.long, device=self.device)
            ))
        
        # 使用模型获取代码向量
        self.model["attention_network"].eval()
        
        with torch.no_grad():
            # 如果没有路径上下文，使用零向量
            if not contexts:
                code_vector = torch.zeros(self.hidden_dim, device=self.device)
            else:
                code_vector = self.model["attention_network"](contexts)
        
        # 转换为NumPy数组并返回
        return code_vector.cpu().numpy()
    
    def save_model(self, path: str):
        """保存模型到文件"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        model_data = {
            "attention_network_state": self.model["attention_network"].state_dict(),
            "classifier_state": self.model["classifier"].state_dict(),
            "token_vocab_size": self.code_vectorizer.next_token_index,
            "path_vocab_size": self.code_vectorizer.next_path_index,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "label_to_index": self.label_to_index,
            "index_to_label": self.index_to_label,
            "token_to_index": self.code_vectorizer.token_to_index,
            "path_to_index": self.code_vectorizer.path_to_index
        }
        
        torch.save(model_data, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """从文件加载模型"""
        model_data = torch.load(path, map_location=self.device)
        
        # 恢复模型参数
        self.embedding_dim = model_data["embedding_dim"]
        self.hidden_dim = model_data["hidden_dim"]
        self.label_to_index = model_data["label_to_index"]
        self.index_to_label = model_data["index_to_label"]
        
        # 恢复向量化器状态
        self.code_vectorizer = CodeVectorizer(self.embedding_dim)
        self.code_vectorizer.token_to_index = model_data["token_to_index"]
        self.code_vectorizer.path_to_index = model_data["path_to_index"]
        self.code_vectorizer.next_token_index = model_data["token_vocab_size"]
        self.code_vectorizer.next_path_index = model_data["path_vocab_size"]
        
        # 恢复模型
        attention_network = PathAttentionNetwork(
            model_data["token_vocab_size"], 
            model_data["path_vocab_size"],
            self.embedding_dim,
            self.hidden_dim
        ).to(self.device)
        
        classifier = nn.Linear(self.hidden_dim, len(self.label_to_index)).to(self.device)
        
        # 加载状态
        attention_network.load_state_dict(model_data["attention_network_state"])
        classifier.load_state_dict(model_data["classifier_state"])
        
        self.model = {
            "attention_network": attention_network,
            "classifier": classifier
        }
        
        print(f"模型已从{path}加载")
    
    def find_similar_codes(self, query_code: str, code_samples: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """找到与查询代码最相似的代码示例"""
        # 获取查询代码的向量
        query_vector = self.get_code_vector(query_code)
        
        # 获取所有代码样本的向量
        sample_vectors = [self.get_code_vector(code) for code in code_samples]
        
        # 计算余弦相似度
        similarities = []
        for i, sample_vector in enumerate(sample_vectors):
            # 计算余弦相似度
            dot_product = np.dot(query_vector, sample_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_sample = np.linalg.norm(sample_vector)
            
            if norm_query == 0 or norm_sample == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_sample)
            
            similarities.append((i, similarity))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# 使用示例
if __name__ == "__main__":
    # 示例代码
    example_code = """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
    """
    
    # 初始化CodeInsight
    insight = Algo2Vec()
    
    # 解析并提取路径
    extractor = PathExtractor()
    paths = extractor.extract_paths(example_code)
    print(f"提取的路径数量: {len(paths)}")
    
    # 获取代码向量
    vectorizer = CodeVectorizer()
    vectorizer.build_vocab([example_code])
    vectorized = vectorizer.vectorize(example_code)
    print(f"向量化后的路径上下文数量: {len(vectorized)}")
    
    print("初始化完成，CodeInsight准备就绪！")