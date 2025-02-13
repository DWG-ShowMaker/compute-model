import random
import torch
from collections import Counter
import torch.nn as nn
import torch.optim as optim

# 生成训练数据
def generate_addition_subtraction_data(num_samples=10000):
    """
    生成加法和减法问题的训练数据。
    
    每个样本包含一个数学表达式（如 "25 + 30"）和对应的答案（以字符串形式）。
    参数:
        num_samples (int): 需要生成的样本数量，默认为 10000。
    返回:
        data (list): 包含 (question, answer) 元组的列表。
    """
    data = []
    for _ in range(num_samples):
        # 随机选择生成加法或减法运算
        num1 = random.randint(0, 100)
        num2 = random.randint(0, 100)
        operation = random.choice(["+", "-"])

        if operation == "+":
            result = num1 + num2
        else:
            result = num1 - num2

        # 构造数学问题字符串和计算得到的答案字符串
        question = f"{num1} {operation} {num2}"
        data.append((question, str(result)))
    return data

# 生成训练数据
training_data = generate_addition_subtraction_data()
print(training_data[:5])  # 打印前5条数据示例

def build_vocab(data):
    # 利用 Counter 统计所有问题和答案中字符的出现频率
    counter = Counter()
    for question, answer in data:
        counter.update(question)
        counter.update(answer)
    
    # 创建一个字典，将每个唯一字符映射到一个唯一的索引
    vocab = {char: idx+2 for idx, (char, _) in enumerate(counter.items())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

# 创建字典
vocab = build_vocab(training_data)

# 显示词汇表的前10个字符映射
print(dict(list(vocab.items())[:10]))

class MathSolver(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MathSolver, self).__init__()
        # 嵌入层：将输入的字符索引转换为固定长度的嵌入向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：接收嵌入向量序列，并提取时序特征，hidden_dim 表示隐藏状态的尺寸
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # 全连接层：将 LSTM 的最后隐藏状态转换为最终输出（数值预测）
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # 前向传播函数:
        # 输入 x 是形状为 [seq_length, batch_size] 的字符索引序列（本例中 batch_size=1）
        
        # 将索引序列转换为嵌入向量，输出形状为 [seq_length, batch_size, embedding_dim]
        embedded = self.embedding(x)
        # 通过 LSTM 层，lstm_out 包含所有时间步的输出，ht 和 ct 为最后的隐藏状态和细胞状态
        lstm_out, (ht, ct) = self.lstm(embedded)
        # 选择最后一个时间步的隐藏状态作为代表，并通过全连接层生成预测数值
        output = self.fc(ht[-1])
        return output

# 模型参数
vocab_size = len(vocab)
embedding_dim = 16
hidden_dim = 128
output_dim = 1  # 结果是一个数字

# 创建模型
model = MathSolver(vocab_size, embedding_dim, hidden_dim, output_dim)

# 显示模型结构
print(model)

def encode_sentence(sentence, vocab):
    # 将句子转换为对应的字符索引列表，并封装为 PyTorch 张量
    return torch.tensor([vocab.get(char, vocab["<UNK>"]) for char in sentence], dtype=torch.long)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 因为输出是一个数字，我们用MSE
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据迭代器
for epoch in range(5):  # 简单训练5个epoch
    total_loss = 0
    for question, answer in training_data:
        optimizer.zero_grad()
        
        # 将数学问题转换为字符索引张量，并增加 batch 维度（此处 batch_size 为1）
        question_tensor = encode_sentence(question, vocab).unsqueeze(1)
        # 将答案转换为浮点数张量
        answer_tensor = torch.tensor([float(answer)], dtype=torch.float)
        
        # 进行前向传播，计算模型预测结果
        output = model(question_tensor)
        
        # 计算本样本的均方误差损失
        loss = criterion(output, answer_tensor)
        # 反向传播，计算梯度
        loss.backward()
        
        # 更新模型参数
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(training_data)}")

# 测试
def predict(question, model, vocab):
    # 将模型设置为评估模式，禁用 dropout 等训练时专用操作
    model.eval()
    with torch.no_grad():
        # 对输入问题进行编码并增加 batch 维度
        question_tensor = encode_sentence(question, vocab).unsqueeze(1)
        output = model(question_tensor)
        # 返回预测的结果
        return output.item()

# 测试示例
test_question = "25 + 30"
prediction = predict(test_question, model, vocab)
print(f"Question: {test_question}, Prediction: {prediction}")