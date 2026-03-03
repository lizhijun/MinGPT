# MinGPT

fork from https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

---

# 费曼学习法：逐行拆解 MicroGPT

> 核心思想：这个文件用 **200 行纯 Python**，从零实现了一个能"学会起名字"的 GPT。没有 PyTorch，没有 TensorFlow，一切从最基本的加减乘除开始。

---

## 第一部分：准备工作（第 1-21 行）

```python
import os, math, random
random.seed(42)
```

**类比**：你要教一个婴儿认字。第一步——找教材。`random.seed(42)` 就像"固定洗牌方式"，确保每次运行程序，数据的顺序完全一样，方便复现结果。

```python
if not os.path.exists('input.txt'):
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
```

**做了什么**：从网上下载一个人名列表（约 3.2 万个英文名字），打乱顺序。每个名字就是一篇"文档"。比如 `["emma", "olivia", "ava", ...]`。

---

## 第二部分：分词器 Tokenizer（第 23-27 行）

```python
uchars = sorted(set(''.join(docs)))  # 所有出现过的字符，如 a-z
BOS = len(uchars)                     # 特殊的"开始/结束"标记
vocab_size = len(uchars) + 1
```

**类比**：想象你有一盒字母积木。把所有名字拆开，收集出现过的**不同字母**，排好序。大概是 `a, b, c, ..., z` 共 26 个（加上少量特殊字符）。

- 每个字母对应一个编号：`a=0, b=1, ..., z=25`
- **BOS**（Begin of Sequence）= 26，是一个特殊积木，表示"名字的开头和结尾"

所以 `"emma"` 变成了 `[26, 4, 12, 12, 0, 26]`（BOS + e + m + m + a + BOS）。

**为什么需要这个？** 计算机不认识字母，只认识数字。分词器就是字母和数字之间的**翻译官**。

---

## 第三部分：自动求导引擎 Autograd（第 29-72 行）

这是整个程序**最精妙**的部分。

### 3.1 Value 类——会"记忆"的数字

```python
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data            # 这个数字的值
        self.grad = 0               # 这个数字对最终损失的"影响力"
        self._children = children   # 谁参与了计算
        self._local_grads = local_grads  # 局部导数
```

**类比**：普通的数字就像一次性纸杯——用完就扔，不记得任何事。而 `Value` 就像一个**带记忆的数字**：

- 它知道自己的**值**是多少（`data`）
- 它知道自己是被**谁**计算出来的（`children`）
- 它知道自己对孩子们的**敏感度**（`local_grads`）

### 3.2 基本运算——搭积木

```python
def __add__(self, other):  # 加法
    return Value(self.data + other.data, (self, other), (1, 1))

def __mul__(self, other):  # 乘法
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

**类比**：想象你在搭乐高。

- `a + b = c`：c 记住了"我是 a 和 b 加出来的"，而且 c 对 a 和 b 的敏感度都是 1（加法的导数）
- `a * b = c`：c 记住了"我是 a 和 b 乘出来的"，c 对 a 的敏感度是 b 的值，对 b 的敏感度是 a 的值（乘法的导数）

每一步运算都会在背后偷偷**画一条线**，把所有运算连成一张**计算图**（就像家谱一样）。

### 3.3 反向传播 backward——追溯责任

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):  # 拓扑排序：先找到所有祖先
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1  # 损失对自己的导数 = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad  # 链式法则！
```

**类比——追查事故责任**：

假设公司亏了钱（loss）。老板想知道：**每个员工对亏损负多大责任？**

1. 先画出公司的组织架构图（`build_topo`——拓扑排序）
2. 从最终的亏损出发，一层一层往下追（`reversed(topo)`）
3. 每经过一个节点，用**链式法则**把"责任"传递下去：
   - `child.grad += local_grad * v.grad`
   - 意思是：**孩子的责任 = 孩子对父亲的影响力 × 父亲的责任**

这就是深度学习最核心的算法：**反向传播（Backpropagation）**。只用 10 行代码实现了！

---

## 第四部分：初始化模型参数（第 74-90 行）

```python
n_layer = 1      # 1层 Transformer
n_embd = 16      # 每个向量16维
block_size = 16   # 最多看16个字符
n_head = 4        # 4个注意力头
```

```python
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) ...]]
state_dict = {
    'wte': matrix(vocab_size, n_embd),   # 词嵌入矩阵
    'wpe': matrix(block_size, n_embd),   # 位置嵌入矩阵
    'lm_head': matrix(vocab_size, n_embd), # 输出层
    ...  # 每层还有 attn_wq, attn_wk, attn_wv, attn_wo, mlp_fc1, mlp_fc2
}
```

**类比**：模型的"知识"存储在一大堆数字里，这些数字叫**参数**。

- 一开始，所有参数都是随机的小数（`random.gauss(0, 0.08)`）——就像一个婴儿的大脑，什么都不懂
- `wte`（word token embedding）：给每个字母一个 16 维的"身份证"
- `wpe`（word position embedding）：给每个位置一个 16 维的"座位号"
- 其他矩阵：注意力机制和前馈网络的"突触连接"

最终统计：`num params` 大约有几千个参数。

---

## 第五部分：模型架构（第 92-144 行）

### 5.1 辅助函数

```python
def linear(x, w):  # 线性变换：矩阵乘法
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

**类比**：把一个向量通过一面"棱镜"，变成另一个向量。每个输出元素是输入的加权求和。

```python
def softmax(logits):  # 把任意数字变成概率分布
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

**类比**：一场选举。每个候选人有一个"原始得分"，softmax 把它们变成"得票率"——所有概率加起来等于 1。减去 max_val 是为了数值稳定（防止指数爆炸）。

```python
def rmsnorm(x):  # 归一化：控制向量的"音量"
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

**类比**：想象一群人同时说话。rmsnorm 就是一个**自动音量调节器**——如果大家都喊得太大声，就把音量调小；太小声就调大。让信号保持在合理范围内。

### 5.2 GPT 核心——gpt() 函数

```python
def gpt(token_id, pos_id, keys, values):
```

这个函数一次处理**一个字符**，输出"下一个字符是什么"的概率。

**第一步：嵌入（Embedding）**

```python
tok_emb = state_dict['wte'][token_id]  # 查字母的"身份证"
pos_emb = state_dict['wpe'][pos_id]    # 查位置的"座位号"
x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 加在一起
x = rmsnorm(x)
```

**类比**：一个学生走进教室。他有两张卡片——"我是谁"和"我坐在哪"。把两张卡片合并，就是他的完整信息。

**第二步：注意力机制（Attention）**

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query: "我在找什么？"
k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key: "我有什么？"
v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value: "我能提供什么？"
```

**类比——图书馆查书**：

- **Query（查询）**：你手里的搜索关键词——"我想找关于猫的书"
- **Key（钥匙）**：每本书封面上的标签——"这本书讲猫"、"这本书讲狗"
- **Value（内容）**：书的实际内容

过程：

1. 用 Query 和每个 Key 做**点乘**（`q_h[j] * k_h[t][j]`），得到**相关性得分**
2. 用 softmax 把得分变成**注意力权重**（哪本书最相关）
3. 用注意力权重对 Value 做**加权求和**（把最相关的书的内容提取出来）

```python
for h in range(n_head):  # 4个头 = 4个不同的"读者"
```

**多头注意力**：就像 4 个人同时去图书馆，每个人关注不同的角度（一个关注元音模式，一个关注辅音搭配……），最后把发现汇总。

**第三步：残差连接**

```python
x = [a + b for a, b in zip(x, x_residual)]
```

**类比**：注意力处理后的结果 + 原始输入。就像"在原来的基础上，加上新发现"。这防止了信息在传递过程中被遗忘（就像给信号加了一条**捷径**）。

**第四步：MLP（前馈神经网络）**

```python
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 放大到 4 倍宽度
x = [xi.relu() for xi in x]                        # 负数变0，正数不变
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])    # 压缩回原来宽度
```

**类比**：注意力是"从别人那里收集信息"，MLP 是"自己消化思考"。先把信息展开（放大 4 倍），用 ReLU 过滤掉不重要的（负数→0），再压缩回来。

**第五步：输出**

```python
logits = linear(x, state_dict['lm_head'])
```

最终把 16 维向量变成 27 维（vocab_size），每个维度代表一个字符的"得分"。

---

## 第六部分：训练循环（第 146-184 行）

### 6.1 优化器 Adam

```python
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # 动量（一阶矩）
v = [0.0] * len(params)  # 速度（二阶矩）
```

**类比——滚球下山**：

- 普通梯度下降：每一步只看脚下的坡度
- **Adam**：球有"惯性"（m）和"对地形的记忆"（v）
  - `m`：记住最近的平均坡度方向（避免来回震荡）
  - `v`：记住坡度的剧烈程度（陡峭的地方走慢点，平坦的地方走快点）

### 6.2 训练的每一步

```python
for step in range(1000):
    # 1. 取一个名字
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

比如 `"emma"` → `[26, 4, 12, 12, 0, 26]`

```python
    # 2. 逐字符预测
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
```

**类比——完形填空**：

| 输入 | 模型猜测 | 正确答案 |
|------|---------|---------|
| BOS  | 下一个是？ | e |
| e    | 下一个是？ | m |
| m    | 下一个是？ | m |
| m    | 下一个是？ | a |
| a    | 下一个是？ | BOS（结束） |

`loss = -log(正确答案的概率)`：如果模型猜对的概率高（比如 0.9），loss 就小（0.1）；猜对概率低（比如 0.01），loss 就大（4.6）。这就是**交叉熵损失**。

```python
    # 3. 反向传播——追溯每个参数的责任
    loss.backward()

    # 4. 更新参数——让模型变得更好
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0  # 梯度清零，准备下一轮
```

**每一步的完整故事**：

1. 给模型看一个名字，让它猜每个字母
2. 计算"猜得有多差"（loss）
3. 反向传播算出"每个参数要负多大责任"（grad）
4. 用 Adam 优化器微调每个参数，让下次猜得更好
5. 重复 1000 次

---

## 第七部分：推理/生成（第 186-200 行）

```python
temperature = 0.5
for sample_idx in range(20):
    token_id = BOS  # 从"开始"标记出发
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:  # 遇到"结束"标记就停
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

**类比——让婴儿说话**：

训练完成后，模型学会了"英文名字长什么样"。现在让它自己编名字：

1. 给它一个"开始"信号（BOS）
2. 它输出下一个字母的概率分布
3. **按概率抽样**选一个字母（不是选概率最高的，而是按概率随机选，增加多样性）
4. 把选出的字母当输入，继续预测下一个
5. 直到它输出"结束"信号（BOS）

**temperature（温度）**：

- `温度 = 0.1`：非常保守，几乎总选概率最高的 → 名字很"正常"但重复
- `温度 = 1.0`：完全按原始概率 → 名字很有"创意"但可能乱码
- `温度 = 0.5`：折中 → 像真名字，但都是**世界上不存在的新名字**

---

## 总结：一张图看懂全流程

```
 名字数据 "emma"
      ↓
 [分词] → [26, 4, 12, 12, 0, 26]
      ↓
 [嵌入] 字母→向量, 位置→向量, 相加
      ↓
 [注意力] 每个字母关注前面所有字母
      ↓
 [MLP] 自己消化思考
      ↓
 [输出层] → 27个概率（下一个字母是谁？）
      ↓
 [损失] -log(正确答案的概率)
      ↓
 [反向传播] 计算每个参数的梯度
      ↓
 [Adam优化] 微调参数
      ↓
 重复1000次 → 模型学会了起名字！
```

**这 200 行代码的伟大之处**：它证明了 GPT 的核心算法其实并不复杂——就是**自动求导 + 注意力机制 + 梯度下降**。所有的 ChatGPT、Claude 背后的基本原理，都在这里了。剩下的"只是效率"——用 GPU 加速、用更多层、用更多数据。
