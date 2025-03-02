# algo2vec

algo2vec is a deep learning-based algorithm and code representation tool that can convert source code into fixed-length vector representations to support various code understanding and analysis tasks. Inspired by word2vec and code2vec, algo2vec focuses on capturing the semantic features of algorithms and code.

## Project Features

- 🔍 **Semantic Code Understanding**: Captures semantic features of code through deep learning, not just surface syntax
- 🔄 **Path Abstraction**: Uses AST paths to represent code structure and relationships
- ⚡ **Attention Mechanism**: Intelligently weights the importance of different paths
- 🔮 **Method Name Prediction**: Infers appropriate method names based on method body
- 🔍 **Code Search**: Based on semantic similarity rather than simple text matching

## Installation

```bash
# Clone repository
git clone https://github.com/amplimit/algo2vec.git
cd algo2vec

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from algo2vec import Algo2Vec

# Initialize
algo2vec = Algo2Vec()

# Analyze code
code = """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""

# Train model
algo2vec.train(code_samples, labels)

# Predict method name
predictions = algo2vec.predict(code, top_k=3)
print(f"Predicted method names: {predictions}")

# Get code vector
code_vector = algo2vec.get_code_vector(code)

# Compare code similarity
similar_codes = algo2vec.find_similar_codes(query_code, code_database, top_k=5)
for idx, similarity in similar_codes:
    print(f"Similarity: {similarity:.2f}, Index: {idx}")
```

## Project Structure

```
algo2vec/
├── algo2vec_core.py      # Core functionality implementation
├── demo.py               # Demo application
├── train.py              # Model training script
├── models/               # Pre-trained models
├── data/                 # Sample data
├── examples/             # Usage examples
├── tests/                # Tests
└── README.md             # Project documentation
```

## Detailed Features

### Code Vectorization

CodeInsight uses an AST path-based representation method to convert code into vectors:

1. Parse source code into an abstract syntax tree (AST)
2. Extract path contexts from the tree, each context includes:
   - Starting terminal value (variable name, type, etc.)
   - AST path
   - Ending terminal value
3. Map path contexts to vector space
4. Aggregate multiple path vectors using attention mechanism
5. Generate a single code vector representation

### Method Name Prediction

By learning the relationship between code structure and function names, CodeInsight can:

- Understand the functional intent of code
- Suggest descriptive method names
- Detect improperly named methods

### Code Similarity Analysis

CodeInsight can calculate semantic similarity between code snippets, which is useful for:

- Code clone detection
- Code search
- Code recommendation
- Code refactoring suggestions

## Training Your Own Model

```python
from algo2vec import Algo2Vec

# Prepare training data
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
    # More code examples...
]
labels = ["isPrime", ...]  # Corresponding label list

# Initialize and train
algo2vec = Algo2Vec(embedding_dim=128, hidden_dim=128)
algo2vec.train(
    code_samples=code_samples,
    labels=labels,
    epochs=20,
    batch_size=32,
    learning_rate=0.001,
    validation_split=0.1
)

# Save model
algo2vec.save_model("my_model.pt")

# Load model
algo2vec.load_model("my_model.pt")
```

## Example Applications

Run the demo application to experience algo2vec's features:

```bash
python demo.py
```

The demo application showcases:
- Model training
- Code analysis
- Method name prediction
- Code syntax and semantic similarity calculation
- Code vector visualization
- Semantic-based code search

## Future Plans

- [ ] Add support for more programming languages such as Java, C++, etc.
- [ ] Improve path extraction algorithm and path representation
- [ ] Integrate Transformer architecture to enhance performance
- [ ] Train pre-trained models on large-scale code repositories
- [ ] Support code generation and auto-completion
- [ ] Develop VSCode plugin and command-line tools
- [ ] Add code refactoring suggestion functionality

## Contributing

Contributions are welcome! Please check the [contribution guidelines](CONTRIBUTING.md) to learn how to participate in the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Feature Comparison

| Feature | algo2vec | code2vec |
|---------|----------|----------|
| Code Representation | AST paths | AST paths |
| Attention Mechanism | Soft attention | Soft attention |
| Deep Learning Framework | PyTorch | TensorFlow |
| Multi-language Support | Python (extensible) | Java |
| Code Similarity | Syntax + Semantics | Not supported |
| Code Search | Supported | Not supported |
| Method Name Prediction | Supported | Supported |
| Online Demo | Local demo | Web demo |
| GPU Acceleration | Supported | Supported |

## Acknowledgments

This project was inspired by [code2vec](https://github.com/tech-srl/code2vec) and [word2vec](https://code.google.com/archive/p/word2vec/). We thank the authors of these projects for their research and code foundations.