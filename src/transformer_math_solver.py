import random
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim

def generate_math_data(num_samples=10000):
    """
    生成一个加减法数学问题数据集，用于训练模型。
    
    每个样本由表示数学问题的字符串（例如 "25 + 30"）和对应答案的字符串组成。
    参数:
        num_samples (int): 生成样本数量，默认为 10000。
    返回:
        data (list): 一个 (question, answer) 元组的列表。
    """
    data = []
    for _ in range(num_samples):
        num1 = random.randint(0, 100)
        num2 = random.randint(0, 100)
        operation = random.choice(["+", "-"])
        result = num1 + num2 if operation == "+" else num1 - num2
        question = f"{num1} {operation} {num2}"
        data.append((question, str(result)))
    return data

# 生成训练数据
training_data = generate_math_data()
print(training_data[:5])  # 查看前5条数据

# 构建词汇表
def build_vocab(data):
    """
    构建字符词汇表，将数据集中所有出现的字符映射为唯一索引。
    
    遍历每个数学问题和答案，为每个字符分配一个唯一的索引，并预留
    "<PAD>" 用于填充，"<UNK>" 用于未知字符。
    """
    counter = Counter()
    for question, answer in data:
        counter.update(question)  # 数学表达式
        counter.update(answer)    # 计算结果
    vocab = {char: idx + 2 for idx, (char, _) in enumerate(counter.items())}
    vocab["<PAD>"] = 0  # 填充字符
    vocab["<UNK>"] = 1  # 未知字符
    return vocab

# 创建字符到索引的映射
vocab = build_vocab(training_data)
print(dict(list(vocab.items())[:10]))  # 查看部分词汇表

def encode_sentence(sentence, vocab):
    """
    将输入句子中的每个字符转换为其对应的索引列表，
    如果字符不在词汇表中，则使用 "<UNK>" 的索引表示。
    """
    return [vocab.get(char, vocab["<UNK>"]) for char in sentence]

class TransformerMathSolver(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers):
        super(TransformerMathSolver, self).__init__()
        # 嵌入层：将词汇索引转换为连续嵌入向量，尺寸为 (vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 位置编码：为序列中每个位置提供固定的向量，形状为 (1, max_seq_length, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 20, embedding_dim))
        
        # Transformer 编码器层：
        # 使用单层 TransformerEncoderLayer 作为构建模块，然后堆叠 num_layers 层组成完整的编码器
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, 
                                                        nhead=num_heads, 
                                                        dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # 输出层：将编码器输出的向量通过全连接层转换为一个预测数值
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        """
        1. 将输入的索引序列通过嵌入层转换为向量，并添加位置编码信息
        embedded 形状: [batch_size, seq_length, embedding_dim]
        2. 将嵌入后的数据输入 Transformer 编码器，进行全局特征提取
        transformer_out 形状同样为 [batch_size, seq_length, embedding_dim]
        3. 对编码器的输出进行平均池化（在序列维度上求均值），得到每个样本的固定长度向量
        pooled 形状: [batch_size, embedding_dim]
        4. 通过全连接层将池化后的向量转换为预测结果（一个数值输出）
        output 形状: [batch_size, 1]
        """
        embedded = self.embedding(x) + self.positional_encoding[:, :x.shape[1], :]
        transformer_out = self.transformer_encoder(embedded)
        pooled = transformer_out.mean(dim=1)
        output = self.fc(pooled)
        return output
    
    # 超参数
embedding_dim = 32
num_heads = 4
hidden_dim = 128
num_layers = 2

# 创建模型
vocab_size = len(vocab)
model = TransformerMathSolver(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练数据转换
def preprocess_data(data, vocab):
    """
    将原始数据集中每个样本的数学问题转换为索引序列，并将答案转换为浮点数
    """
    X, Y = [], []
    for question, answer in data:
        X.append(encode_sentence(question, vocab))
        Y.append(float(answer))
    return X, Y

X_train, Y_train = preprocess_data(training_data, vocab)

# 填充序列（保证相同长度）
import torch.nn.functional as F

def pad_sequences(sequences, max_length, pad_value=0):
    """
    对于每个序列，在右侧补充 pad_value 直到序列长度为 max_length
    """
    return [seq + [pad_value] * (max_length - len(seq)) for seq in sequences]

max_length = max(len(seq) for seq in X_train)
X_train = pad_sequences(X_train, max_length)
X_train = torch.tensor(X_train, dtype=torch.long)
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

# 训练循环
num_epochs = 5
batch_size = 32

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_Y = Y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train)}")

# 测试
def predict(question, model, vocab):
    """
    设置模型为评估模式，禁用 dropout 等训练时专用操作
    """
    model.eval()
    with torch.no_grad():
        question_tensor = torch.tensor(pad_sequences([encode_sentence(question, vocab)], max_length), dtype=torch.long)
        output = model(question_tensor)
        return round(output.item(), 2)

# 测试示例
test_questions = ["25 + 30", "100 - 50", "45 + 20", "10 + 90"]
for q in test_questions:
    print(f"Question: {q}, Prediction: {predict(q, model, vocab)}")