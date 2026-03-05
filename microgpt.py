"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

用纯 Python（零依赖）实现 GPT 的训练和推理。
本文件即完整算法，其余一切皆为效率优化。

@karpathy
"""

import os       # os.path.exists — 检查文件是否存在
import math     # math.log, math.exp — 对数和指数运算
import random   # random.seed, random.choices, random.gauss, random.shuffle — 随机数工具
random.seed(42) # 固定随机种子，保证每次运行结果可复现（Let there be order among chaos）

# ============================================================================
# 第一部分：数据集加载
# 数据集 `docs` 是一个字符串列表，每个字符串是一个人名（如 "Emma", "Olivia"）
# 数据来自 Karpathy 的 makemore 项目，共约 32K 个英文人名
# ============================================================================
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')  # 自动下载数据文件
docs = [line.strip() for line in open('input.txt') if line.strip()]  # 读取并过滤空行
random.shuffle(docs)  # 随机打乱顺序，防止训练时产生顺序偏差
print(f"num docs: {len(docs)}")

# ============================================================================
# 第二部分：Tokenizer（分词器）
# 将字符串转换为整数序列（"token"），再从整数序列还原为字符串
# 这里使用最简单的字符级分词：每个唯一字符对应一个 token id
# ============================================================================
uchars = sorted(set(''.join(docs)))  # 收集数据集中所有唯一字符，排序后得到字符表（如 a-z）
BOS = len(uchars)      # BOS（Beginning of Sequence）特殊 token，用于标记序列的开头和结尾
                        # 它的 id 刚好是字符数量（紧接在字符 id 之后）
vocab_size = len(uchars) + 1  # 词汇表大小 = 字符数 + 1（BOS token）
print(f"vocab size: {vocab_size}")

# ============================================================================
# 第三部分：自动微分引擎（Autograd）
# 这是整个系统的核心基础设施。Value 类包装了一个标量值，并记录计算图，
# 从而在 backward() 时能通过链式法则（chain rule）自动计算梯度。
#
# 原理：每次运算（如 a + b）都会创建新的 Value 节点，记录：
#   1. 前向计算的结果（data）
#   2. 参与运算的子节点（children）
#   3. 输出对每个子节点的局部导数（local_grads）
# 反向传播时，从 loss 节点出发，沿计算图反向传递梯度。
# ============================================================================
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    # __slots__ 是 Python 优化技巧：禁用 __dict__，减少每个实例的内存开销
    # 在有数百万个 Value 节点的计算图中，这一优化效果显著

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 该节点的标量值（前向传播时计算得到）
        self.grad = 0                   # 损失函数对该节点的导数（反向传播时累加得到）
        self._children = children       # 该节点在计算图中的子节点（即参与运算的输入）
        self._local_grads = local_grads # 该节点输出对每个子节点的局部导数
                                        # 例如 c = a + b 时，dc/da = 1, dc/db = 1

    # --- 加法：c = a + b ---
    # 局部导数：dc/da = 1, dc/db = 1
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    # --- 乘法：c = a * b ---
    # 局部导数：dc/da = b.data, dc/db = a.data（交叉相乘）
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    # --- 幂运算：c = a^n ---
    # 局部导数：dc/da = n * a^(n-1)（幂函数求导法则）
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))

    # --- 对数：c = ln(a) ---
    # 局部导数：dc/da = 1/a
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))

    # --- 指数：c = e^a ---
    # 局部导数：dc/da = e^a（指数函数的导数等于自身）
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    # --- ReLU 激活函数：c = max(0, a) ---
    # 局部导数：a > 0 时为 1，否则为 0（分段线性函数）
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # --- 以下是运算符的语法糖，全部基于上面的基础运算实现 ---
    def __neg__(self): return self * -1                  # -a = a * (-1)
    def __radd__(self, other): return self + other       # 支持 int/float + Value
    def __sub__(self, other): return self + (-other)     # a - b = a + (-b)
    def __rsub__(self, other): return other + (-self)    # 支持 int/float - Value
    def __rmul__(self, other): return self * other       # 支持 int/float * Value
    def __truediv__(self, other): return self * other**-1   # a / b = a * b^(-1)
    def __rtruediv__(self, other): return other * self**-1  # 支持 int/float / Value

    def backward(self):
        """
        反向传播：从当前节点（通常是 loss）出发，计算损失对所有节点的梯度。

        步骤：
        1. 拓扑排序：用 DFS 后序遍历得到计算图的拓扑序列。
           保证处理每个节点时，所有依赖它的节点都已处理完毕。
        2. 设置 loss 节点的梯度为 1（dL/dL = 1）。
        3. 按拓扑逆序遍历，应用链式法则：
           child.grad += local_grad * parent.grad
           即：子节点梯度 += 局部导数 × 父节点梯度
        """
        topo = []         # 拓扑排序结果
        visited = set()   # 已访问节点集合，防止重复遍历
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)  # 先递归处理子节点
                topo.append(v)        # 后序添加当前节点
        build_topo(self)
        self.grad = 1  # 起点：dL/dL = 1
        for v in reversed(topo):  # 逆拓扑序 = 从输出到输入
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
                # 链式法则的精髓：梯度 = 局部导数 × 上游传来的梯度
                # 用 += 而非 = 是因为一个节点可能被多条路径使用（梯度需要累加）

# ============================================================================
# 第四部分：模型参数初始化
# GPT 的所有"知识"都存储在这些参数矩阵中。
# 训练前随机初始化，训练过程中通过梯度下降不断优化。
# ============================================================================
n_layer = 1      # Transformer 层数（深度），这里只用 1 层，极简版
n_embd = 16      # 嵌入维度（网络宽度），每个 token 用 16 维向量表示
block_size = 16   # 最大上下文长度（注意力窗口大小），最长人名为 15 个字符，所以 16 刚好够用
n_head = 4        # 多头注意力的头数，每个头独立关注不同的特征模式
head_dim = n_embd // n_head  # 每个注意力头的维度 = 16 / 4 = 4

# 参数矩阵生成函数：创建 nout × nin 的随机矩阵
# std=0.08 是初始化标准差，较小的值有助于训练初期的稳定性
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# state_dict：模型的参数字典，命名风格与 PyTorch 一致
state_dict = {
    'wte': matrix(vocab_size, n_embd),    # Token 嵌入矩阵 (vocab_size × n_embd)
                                           # 每行是一个 token 的向量表示
    'wpe': matrix(block_size, n_embd),     # 位置嵌入矩阵 (block_size × n_embd)
                                           # 编码每个位置的信息（第1个字符、第2个字符...）
    'lm_head': matrix(vocab_size, n_embd), # 语言模型头 (vocab_size × n_embd)
                                           # 将隐藏状态映射回词汇表大小，用于预测下一个 token
}
for i in range(n_layer):
    # 每一层 Transformer 包含两个子模块：注意力 + MLP
    # --- 注意力子模块的 4 个权重矩阵 ---
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query 投影：将输入映射为"查询"
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key 投影：将输入映射为"键"
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value 投影：将输入映射为"值"
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # Output 投影：合并多头注意力的输出
    # --- MLP 子模块的 2 个权重矩阵 ---
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # 第一层全连接：升维到 4 倍（64 维）
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # 第二层全连接：降维回原始维度（16 维）

# 将所有参数矩阵展平为一维列表，方便优化器统一处理
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ============================================================================
# 第五部分：模型架构
# 遵循 GPT-2 架构，但做了三处简化：
#   1. LayerNorm → RMSNorm（更简单，无均值中心化）
#   2. 无偏置项（所有线性层都只有权重矩阵）
#   3. GeLU → ReLU（更简单的激活函数）
# ============================================================================

def linear(x, w):
    """
    线性变换（矩阵乘法）：y = xW^T
    x: 输入向量 [n_in]，每个元素是 Value
    w: 权重矩阵 [n_out × n_in]，每个元素是 Value
    返回: 输出向量 [n_out]

    这是神经网络最基础的运算单元。每个输出元素 = 对应权重行与输入的点积。
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """
    Softmax 函数：将任意实数向量转换为概率分布（所有元素非负且和为 1）。
    公式：softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

    减去 max(x) 是数值稳定性技巧：防止 exp() 溢出（不影响数学结果，因为分子分母同时缩放）。
    在 GPT 中用于：
      1. 注意力权重的归一化
      2. 最终输出的概率分布
    """
    max_val = max(val.data for val in logits)           # 找到最大值（用 .data 因为这只是为了数值稳定）
    exps = [(val - max_val).exp() for val in logits]    # 减去最大值后取指数
    total = sum(exps)                                   # 求和（作为归一化分母）
    return [e / total for e in exps]                    # 每个元素除以总和，得到概率

def rmsnorm(x):
    """
    RMSNorm（Root Mean Square Normalization）：将向量归一化到单位 RMS 长度。
    公式：rmsnorm(x) = x / sqrt(mean(x^2) + ε)

    与 LayerNorm 的区别：不做均值中心化（减去均值），只做缩放。
    ε = 1e-5 防止除以零。
    作用：稳定训练过程，防止各层激活值的尺度不断膨胀或缩小。
    """
    ms = sum(xi * xi for xi in x) / len(x)  # 均方值 (mean square)
    scale = (ms + 1e-5) ** -0.5              # 缩放因子 = 1 / sqrt(ms + ε)
    return [xi * scale for xi in x]          # 每个元素乘以缩放因子

def gpt(token_id, pos_id, keys, values):
    """
    GPT 前向传播：给定一个 token 及其位置，输出下一个 token 的 logits（未归一化的概率）。

    参数:
      token_id: 当前 token 的 id
      pos_id:   当前 token 在序列中的位置索引
      keys:     KV 缓存中的 Key，形状 [n_layer][已处理的位置数][n_embd]
      values:   KV 缓存中的 Value，形状同上

    KV 缓存机制：每次只处理一个 token，但将其 K、V 追加到缓存中，
    使注意力能看到之前所有位置的信息。这也是 GPT 的自回归生成方式。

    返回: logits 向量 [vocab_size]，表示下一个 token 的预测分数
    """
    # --- 嵌入层 ---
    tok_emb = state_dict['wte'][token_id]  # 从嵌入矩阵查表，得到 token 的向量表示
    pos_emb = state_dict['wpe'][pos_id]    # 从位置矩阵查表，得到位置编码
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 将 token 嵌入和位置嵌入逐元素相加
    x = rmsnorm(x)  # 虽然看起来多余（残差连接前就归一化），但因为反向传播经过残差连接，
                     # 这里的归一化实际上影响了梯度流，对训练有帮助

    for li in range(n_layer):
        # ==================== 1) 多头注意力子模块 ====================
        # 注意力机制的核心思想：让每个 token "关注" 序列中其他 token 的信息
        x_residual = x  # 保存残差连接的输入（稍后会加回来）
        x = rmsnorm(x)  # Pre-Norm：在注意力计算前做归一化

        # 通过三个线性投影，将输入 x 分别映射为 Query、Key、Value
        # Q: "我在找什么？"    K: "我是什么？"    V: "我能提供什么信息？"
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query 投影 [n_embd]
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key 投影 [n_embd]
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value 投影 [n_embd]

        # 将当前位置的 K、V 追加到 KV 缓存（自回归的关键）
        keys[li].append(k)
        values[li].append(v)

        # 多头注意力：将 Q、K、V 按维度拆分为 n_head 个独立的注意力头
        # 每个头只处理 head_dim = 4 个维度，独立计算注意力
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim  # 当前头的起始维度索引

            # 取出当前头对应的 Q、K、V 切片
            q_h = q[hs:hs+head_dim]                            # 当前 token 的 Query [head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]      # 所有已缓存位置的 Key [t × head_dim]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]    # 所有已缓存位置的 Value [t × head_dim]

            # 计算注意力分数：Q 和每个位置的 K 做点积，再除以 sqrt(head_dim) 进行缩放
            # 缩放的原因：随着维度增大，点积的方差也增大，缩放后使 softmax 的输入保持合理范围
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]

            # 注意力权重：通过 softmax 将分数转换为概率分布
            attn_weights = softmax(attn_logits)

            # 加权聚合 Value：注意力权重高的位置贡献更多信息
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)  # 将各头输出拼接

        # Output 投影：将多头注意力的拼接结果映射回原始维度
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])

        # 残差连接：将注意力输出加上输入，缓解深层网络的梯度消失问题
        x = [a + b for a, b in zip(x, x_residual)]

        # ==================== 2) MLP 子模块（前馈网络） ====================
        # 两层全连接网络，中间用 ReLU 激活函数引入非线性
        # 先升维到 4 倍（增加表达能力），再降维回原始维度
        x_residual = x
        x = rmsnorm(x)  # Pre-Norm
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 升维：16 → 64
        x = [xi.relu() for xi in x]                       # ReLU 激活：max(0, x)，引入非线性
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 降维：64 → 16
        x = [a + b for a, b in zip(x, x_residual)]       # 残差连接

    # 语言模型头：将最终隐藏状态映射到词汇表大小，得到每个 token 的预测分数
    logits = linear(x, state_dict['lm_head'])
    return logits

# ============================================================================
# 第六部分：Adam 优化器
# Adam（Adaptive Moment Estimation）是深度学习中最常用的优化器。
# 它为每个参数维护两个动量缓冲区：
#   m: 一阶矩（梯度的指数移动平均）— 相当于"平均方向"
#   v: 二阶矩（梯度平方的指数移动平均）— 相当于"平均步幅"
# 结合两者，Adam 能自适应地调整每个参数的更新步长。
# ============================================================================
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
# learning_rate: 基础学习率（控制每步更新的幅度）
# beta1 = 0.85: 一阶矩的衰减率（通常用 0.9，这里稍低以加快对新梯度的响应）
# beta2 = 0.99: 二阶矩的衰减率
# eps_adam = 1e-8: 防止除以零的小常数

m = [0.0] * len(params)  # 一阶矩缓冲（梯度均值的估计）
v = [0.0] * len(params)  # 二阶矩缓冲（梯度方差的估计）

# ============================================================================
# 第七部分：训练循环
# 核心流程：前向传播 → 计算损失 → 反向传播 → 参数更新
# 每一步取一个文档（人名），让模型学习预测下一个字符。
# 经过 1000 步训练，模型应该能学会英文人名的统计规律。
# ============================================================================
num_steps = 1000  # 训练步数
for step in range(num_steps):

    # --- 数据准备 ---
    # 取出一个人名，转换为 token 序列，两端加 BOS 标记
    # 例如 "emma" → [BOS, e, m, m, a, BOS]
    # BOS 在开头表示"序列开始"，在结尾表示"序列结束"
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)  # 实际处理的 token 数（不超过上下文窗口）

    # --- 前向传播：构建计算图并计算损失 ---
    # KV 缓存在每个新文档开始时清空
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        # 给模型看 token_id，期望它预测 target_id
        logits = gpt(token_id, pos_id, keys, values)  # 前向传播，得到预测 logits
        probs = softmax(logits)                         # 转换为概率分布
        loss_t = -probs[target_id].log()                # 交叉熵损失 = -log(正确类别的概率)
                                                         # 概率越接近 1，损失越接近 0
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)  # 对序列中所有位置的损失取平均

    # --- 反向传播：计算所有参数的梯度 ---
    loss.backward()  # 自动微分引擎递归计算 dL/dp 对每个参数 p

    # --- Adam 优化器更新 ---
    lr_t = learning_rate * (1 - step / num_steps)  # 线性学习率衰减：从 0.01 线性降到 0
                                                    # 训练后期用更小的步长，有助于收敛到更好的解
    for i, p in enumerate(params):
        # 更新一阶矩（梯度的指数移动平均）
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        # 更新二阶矩（梯度平方的指数移动平均）
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        # 偏差修正：训练初期 m 和 v 被初始化为 0，会偏向于 0，需要修正
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        # 参数更新：沿梯度方向移动，步长由 Adam 自适应调整
        # 除以 sqrt(v_hat) 的效果：梯度波动大的参数走小步，梯度稳定的参数走大步
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0  # 清零梯度，为下一步做准备（否则梯度会累加）

    # \r 是回车符，覆盖同一行输出，实现进度条效果
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# ============================================================================
# 第八部分：推理（生成新名字）
# 训练完成后，模型已经学会了英文人名的统计规律。
# 现在让它自回归地生成全新的、从未见过的名字。
#
# 生成过程：
#   1. 从 BOS token 开始
#   2. 将当前 token 送入模型，得到下一个 token 的概率分布
#   3. 从概率分布中采样（不是取最大值，而是随机采样，增加多样性）
#   4. 重复直到生成 BOS（表示结束）或达到最大长度
# ============================================================================
temperature = 0.5  # 温度参数 ∈ (0, 1]，控制生成文本的"创造力"
                    # 低温（→0）：倾向于选择概率最高的 token，生成更保守
                    # 高温（→1）：概率分布更均匀，生成更随机、更有创意
                    # 实现方式：将 logits 除以 temperature 后再 softmax

print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    # 每个新名字独立生成，KV 缓存重新初始化
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS        # 从 BOS 开始
    sample = []           # 收集生成的字符
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)           # 前向传播
        probs = softmax([l / temperature for l in logits])      # 除以温度后取 softmax
        # random.choices: 按概率权重采样，等价于从概率分布中抽取一个 token
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:  # 生成了 BOS 表示名字结束
            break
        sample.append(uchars[token_id])  # 将 token id 转回字符
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
