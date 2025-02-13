import random
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def generate_math_data(num_samples=10000):
    """
    生成一个加减法数据集。
    
    每个样本由一个数学问题（例如 "25 + 30"）和对应的答案（字符串形式）组成。
    参数:
        num_samples (int): 需要生成的样本数量，默认为 10000。
    返回:
        data (list): 包含 (question, answer) 元组的列表。
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

# 生成数据
training_data = generate_math_data()
print(training_data[:5])  # 查看前5条数据

def build_vocab(data):
    """
    构建字符词汇表，将数据集中出现的每个字符映射为唯一的索引。
    
    遍历所有问题与答案中的字符，并为每个不同字符分配一个索引。
    同时预留 "<PAD>" 作为填充符号和 "<UNK>" 代表未知字符。
    
    参数:
        data (list): 包含 (question, answer) 元组的数据集。
    返回:
        vocab (dict): 字符到索引的字典映射。
    """
    counter = Counter()
    for question, answer in data:
        counter.update(question)
        counter.update(answer)
    vocab = {char: idx + 2 for idx, (char, _) in enumerate(counter.items())}
    vocab["<PAD>"] = 0  # 填充
    vocab["<UNK>"] = 1  # 未知字符
    return vocab

# 生成词汇表
vocab = build_vocab(training_data)
print(dict(list(vocab.items())[:10]))  # 查看部分字典

class MoETransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, num_experts):
        super(MoETransformer, self).__init__()
        # nn.Embedding: 将每个词汇索引映射到一个连续的嵌入向量表示
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 位置编码: 为每个输入位置提供固定的向量，用于捕捉序列中元素的位置信息
        self.positional_encoding = nn.Parameter(torch.zeros(1, 20, embedding_dim))

        # 门控网络 (Gating Network):
        # 根据整体句子特征决定每个专家的贡献权重，输出维度为专家数量 num_experts
        self.gating_network = nn.Linear(embedding_dim, num_experts)

        # 专家网络 (Experts):
        # 使用多个 TransformerEncoderLayer 作为专家，每个专家独立处理输入的嵌入
        self.experts = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim)
            for _ in range(num_experts)
        ])

        # 输出层:
        # 对融合后的专家输出进行线性变换，最终预测一个数值结果
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # 1. 嵌入查找与位置编码:
        # 将输入的字符索引转换为向量表示，并加上相应的位置信息。
        # 结果 shape: [batch_size, seq_length, embedding_dim]
        embedded = self.embedding(x) + self.positional_encoding[:, :x.shape[1], :]

        # 2. 计算门控权重:
        # 先对嵌入进行平均池化（沿序列维度）得到整体句子表示，
        # 再通过门控网络计算每个专家的得分，使用 softmax 转换为概率分布。
        # 最终 gating_weights 形状为 [batch_size, num_experts]
        gating_weights = F.softmax(self.gating_network(embedded.mean(dim=1)), dim=-1)

        # 3. 利用所有专家网络进行特征提取:
        # 对每个专家，使用相同的嵌入输入进行前向计算，
        # 在专家内部，输出序列会进行 Transformer 处理，
        # 最后沿序列维度进行平均池化，得到每个样本的向量表示。
        # 将所有专家的输出堆叠，形状变为 [batch_size, num_experts, embedding_dim]
        all_expert_outputs = []
        for expert in self.experts:
            expert_out = expert(embedded)
            expert_out = expert_out.mean(dim=1)  # 平均池化得到单个向量表示
            all_expert_outputs.append(expert_out)
        all_expert_outputs = torch.stack(all_expert_outputs, dim=1)

        # 4. 选择得分最高的前 top_k 个专家:
        # 使用 torch.topk 选择每个样本中得分最高的 top_k 个专家，
        # gating_top 包含这 top_k 专家的权重，top_indices 表示对应专家的索引。
        top_k = 2
        gating_top, top_indices = torch.topk(gating_weights, top_k, dim=-1)

        # 5. 根据 top_indices 从所有专家的输出中选取对应的向量：
        # 通过 unsqueeze 和 expand 将 top_indices 转换至匹配 all_expert_outputs 的最后一维，
        # 并使用 torch.gather 收集每个样本中 top_k 专家的输出向量，
        # 得到 top_expert_outputs 形状 [batch_size, top_k, embedding_dim]
        top_expert_outputs = torch.gather(
            all_expert_outputs, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, all_expert_outputs.size(-1))
        )

        # 6. 对选中的专家输出按其对应的权重进行加权求和，
        # gating_top 的 shape 为 [batch_size, top_k]，经过 unsqueeze 后与 top_expert_outputs 相乘，
        # 然后在专家维度上求和得到融合后的特征表示，[batch_size, embedding_dim]
        weighted_output = (top_expert_outputs * gating_top.unsqueeze(-1)).sum(dim=1)

        # 7. 通过全连接层将融合特征转换为最终预测结果（输出为单一数值）
        return self.fc(weighted_output)
    
# 根据是否支持 MPS (Apple M1/M2 加速) 自动选择计算设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 以下是模型超参数配置
embedding_dim = 32
num_heads = 4
hidden_dim = 128
num_layers = 2
num_experts = 4  # 4个专家

# 根据词汇表大小初始化 MoETransformer 模型，并移至所选计算设备（CPU 或 GPU/MPS）
vocab_size = len(vocab)
model = MoETransformer(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, num_experts).to(device)

# 定义损失函数与优化器:
# 使用均方误差 (MSELoss) 作为损失函数，Adam 优化器用于更新模型参数。
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据预处理:
# 将文本问题转换为数字索引表示，并补齐填充以形成等长序列。
def encode_sentence(sentence, vocab):
    # 将句子中每个字符根据词汇字典转换为相应的索引，
    # 如果字符不在字典中，则使用 "<UNK>" 的索引表示
    return [vocab.get(char, vocab["<UNK>"]) for char in sentence]

X_train, Y_train = [], []
for question, answer in training_data:
    X_train.append(encode_sentence(question, vocab))
    Y_train.append(float(answer))

# 对每个序列进行右侧填充，使其长度达到 max_length
# pad_value 默认为 0，对应于 "<PAD>" 索引
def pad_sequences(sequences, max_length, pad_value=0):
    return [seq + [pad_value] * (max_length - len(seq)) for seq in sequences]

max_length = max(len(seq) for seq in X_train)
X_train = torch.tensor(pad_sequences(X_train, max_length), dtype=torch.long).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1).to(device)

# 模型训练:
# 对所有训练数据进行多轮训练，每个 epoch 分批处理数据，
# 并使用反向传播更新模型参数。
num_epochs = 5
batch_size = 32

for epoch in range(num_epochs):
    # 按批处理训练数据，每个批次大小为 batch_size
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        # 1. 清除上一批次的累计梯度
        optimizer.zero_grad()
        batch_X = X_train[i:i+batch_size]
        batch_Y = Y_train[i:i+batch_size]
        
        # 2. 模型前向传播：计算预测结果
        predictions = model(batch_X)
        # 3. 计算当前批次的损失
        loss = criterion(predictions, batch_Y)
        # 4. 反向传播：计算梯度
        loss.backward()
        # 5. 更新模型参数
        optimizer.step()

        total_loss += loss.item()

    # 打印本轮训练的平均损失（总损失除以训练样本数）
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train)}")

# 切换模型到评估模式，关闭 Dropout 等训练特有操作
def predict(question, model, vocab):
    model.eval()
    with torch.no_grad():
        # 对输入问题进行编码和填充，转换为张量
        question_tensor = torch.tensor(
            pad_sequences([encode_sentence(question, vocab)], max_length),
            dtype=torch.long
        ).to(device)
        # 模型前向传播得到预测输出
        output = model(question_tensor)
        # 返回经过四舍五入处理后的预测结果
        return round(output.item(), 2)

# 测试示例
test_questions = ["25 + 30", "100 - 50", "45 + 20", "10 + 90"]
for q in test_questions:
    print(f"Question: {q}, Prediction: {predict(q, model, vocab)}")