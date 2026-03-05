/**
 * The most atomic way to train and run inference for a GPT in pure, dependency-free TypeScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * Translated from @karpathy's microgpt.py
 * Run: npx tsx microgpt.ts
 */

import { readFileSync, writeFileSync, existsSync } from "fs";
import { get } from "https";

// ============================================================================
// 伪随机数生成器（Mulberry32），替代 Python 的 random 模块
// 使用固定种子 42，保证每次运行结果可复现
// ============================================================================
class RNG {
  private state: number;
  constructor(seed: number) {
    this.state = seed;
  }
  /** 返回 [0, 1) 的均匀分布随机数 */
  random(): number {
    this.state |= 0;
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  /** Box-Muller 变换：生成标准正态分布随机数 */
  gauss(mean: number = 0, std: number = 1): number {
    const u1 = this.random();
    const u2 = this.random();
    return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  /** Fisher-Yates 洗牌算法 */
  shuffle<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
}

const rng = new RNG(42);

// ============================================================================
// 自动微分引擎（Autograd）
// Value 类包装标量值并记录计算图，backward() 通过链式法则自动计算梯度
// ============================================================================
class Value {
  data: number;
  grad: number = 0;
  private children: Value[];
  private localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this.children = children;
    this.localGrads = localGrads;
  }

  add(other: Value): Value {
    return new Value(this.data + other.data, [this, other], [1, 1]);
  }
  addS(s: number): Value {
    return this.add(new Value(s));
  }
  mul(other: Value): Value {
    return new Value(this.data * other.data, [this, other], [other.data, this.data]);
  }
  mulS(s: number): Value {
    return this.mul(new Value(s));
  }
  pow(n: number): Value {
    return new Value(this.data ** n, [this], [n * this.data ** (n - 1)]);
  }
  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }
  exp(): Value {
    const e = Math.exp(this.data);
    return new Value(e, [this], [e]);
  }
  relu(): Value {
    const d = this.data > 0 ? this.data : 0;
    const g = this.data > 0 ? 1 : 0;
    return new Value(d, [this], [g]);
  }
  neg(): Value {
    return this.mulS(-1);
  }
  sub(other: Value): Value {
    return this.add(other.neg());
  }
  div(other: Value): Value {
    return this.mul(other.pow(-1));
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const buildTopo = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v.children) buildTopo(child);
      topo.push(v);
    };
    buildTopo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v.children.length; j++) {
        v.children[j].grad += v.localGrads[j] * v.grad;
      }
    }
  }
}

// ============================================================================
// 辅助函数
// ============================================================================
function sumValues(vals: Value[]): Value {
  let s = new Value(0);
  for (const v of vals) s = s.add(v);
  return s;
}

function linear(x: Value[], w: Value[][]): Value[] {
  return w.map((wo) => {
    let s = new Value(0);
    for (let j = 0; j < x.length; j++) s = s.add(wo[j].mul(x[j]));
    return s;
  });
}

function softmax(logits: Value[]): Value[] {
  let maxVal = logits[0].data;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i].data > maxVal) maxVal = logits[i].data;
  }
  const exps = logits.map((v) => v.addS(-maxVal).exp());
  const total = sumValues(exps);
  return exps.map((e) => e.div(total));
}

function rmsnorm(x: Value[]): Value[] {
  let s = new Value(0);
  for (const xi of x) s = s.add(xi.mul(xi));
  const ms = s.mulS(1 / x.length);
  const scale = ms.addS(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

// ============================================================================
// 下载数据集 & 启动训练
// ============================================================================
function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = require("fs").createWriteStream(dest);
    get(url, (resp) => {
      // handle redirects
      if (resp.statusCode === 301 || resp.statusCode === 302) {
        file.close();
        downloadFile(resp.headers.location!, dest).then(resolve).catch(reject);
        return;
      }
      resp.pipe(file);
      file.on("finish", () => { file.close(); resolve(); });
    }).on("error", reject);
  });
}

async function main() {
  // --- 数据集加载 ---
  if (!existsSync("input.txt")) {
    console.log("Downloading input.txt...");
    await downloadFile(
      "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt",
      "input.txt"
    );
  }
  const docs = readFileSync("input.txt", "utf-8")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  rng.shuffle(docs);
  console.log(`num docs: ${docs.length}`);

  // --- Tokenizer ---
  const charSet = new Set<string>();
  for (const doc of docs) for (const ch of doc) charSet.add(ch);
  const uchars = [...charSet].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;
  console.log(`vocab size: ${vocabSize}`);

  const charIndex = new Map<string, number>();
  uchars.forEach((ch, i) => charIndex.set(ch, i));

  // --- 模型参数初始化 ---
  const nLayer = 1;
  const nEmbd = 16;
  const blockSize = 16;
  const nHead = 4;
  const headDim = nEmbd / nHead;

  const matrix = (nout: number, nin: number, std = 0.08): Value[][] => {
    const m: Value[][] = [];
    for (let i = 0; i < nout; i++) {
      const row: Value[] = [];
      for (let j = 0; j < nin; j++) row.push(new Value(rng.gauss(0, std)));
      m.push(row);
    }
    return m;
  };

  const stateDict: Record<string, Value[][]> = {
    wte: matrix(vocabSize, nEmbd),
    wpe: matrix(blockSize, nEmbd),
    lm_head: matrix(vocabSize, nEmbd),
  };
  for (let i = 0; i < nLayer; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(nEmbd, nEmbd);
    stateDict[`layer${i}.attn_wk`] = matrix(nEmbd, nEmbd);
    stateDict[`layer${i}.attn_wv`] = matrix(nEmbd, nEmbd);
    stateDict[`layer${i}.attn_wo`] = matrix(nEmbd, nEmbd);
    stateDict[`layer${i}.mlp_fc1`] = matrix(4 * nEmbd, nEmbd);
    stateDict[`layer${i}.mlp_fc2`] = matrix(nEmbd, 4 * nEmbd);
  }

  // 保持与 Python dict 插入顺序一致
  const paramKeys = ["wte", "wpe", "lm_head"];
  for (let i = 0; i < nLayer; i++) {
    paramKeys.push(
      `layer${i}.attn_wq`, `layer${i}.attn_wk`, `layer${i}.attn_wv`,
      `layer${i}.attn_wo`, `layer${i}.mlp_fc1`, `layer${i}.mlp_fc2`
    );
  }
  const params: Value[] = [];
  for (const key of paramKeys) {
    for (const row of stateDict[key]) params.push(...row);
  }
  console.log(`num params: ${params.length}`);

  // --- GPT 前向传播 ---
  const gpt = (tokenId: number, posId: number, keys: Value[][][], values: Value[][][]): Value[] => {
    const tokEmb = stateDict["wte"][tokenId];
    const posEmb = stateDict["wpe"][posId];
    let x = tokEmb.map((t, i) => t.add(posEmb[i]));
    x = rmsnorm(x);

    for (let li = 0; li < nLayer; li++) {
      // 1) Multi-head Attention
      const xResidual1 = x;
      x = rmsnorm(x);
      const q = linear(x, stateDict[`layer${li}.attn_wq`]);
      const k = linear(x, stateDict[`layer${li}.attn_wk`]);
      const v = linear(x, stateDict[`layer${li}.attn_wv`]);
      keys[li].push(k);
      values[li].push(v);
      const xAttn: Value[] = [];
      for (let h = 0; h < nHead; h++) {
        const hs = h * headDim;
        const qH = q.slice(hs, hs + headDim);
        const kH = keys[li].map((ki) => ki.slice(hs, hs + headDim));
        const vH = values[li].map((vi) => vi.slice(hs, hs + headDim));
        const scale = 1 / Math.sqrt(headDim);
        const attnLogits: Value[] = [];
        for (let t = 0; t < kH.length; t++) {
          let s = new Value(0);
          for (let j = 0; j < headDim; j++) s = s.add(qH[j].mul(kH[t][j]));
          attnLogits.push(s.mulS(scale));
        }
        const attnWeights = softmax(attnLogits);
        for (let j = 0; j < headDim; j++) {
          let s = new Value(0);
          for (let t = 0; t < vH.length; t++) s = s.add(attnWeights[t].mul(vH[t][j]));
          xAttn.push(s);
        }
      }
      x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
      x = x.map((a, i) => a.add(xResidual1[i]));

      // 2) MLP
      const xResidual2 = x;
      x = rmsnorm(x);
      x = linear(x, stateDict[`layer${li}.mlp_fc1`]);
      x = x.map((xi) => xi.relu());
      x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
      x = x.map((a, i) => a.add(xResidual2[i]));
    }

    return linear(x, stateDict["lm_head"]);
  };

  // --- Adam 优化器 ---
  const learningRate = 0.01, beta1 = 0.85, beta2 = 0.99, epsAdam = 1e-8;
  const mBuf = new Float64Array(params.length);
  const vBuf = new Float64Array(params.length);

  // --- 训练循环 ---
  const numSteps = 1000;
  for (let step = 0; step < numSteps; step++) {
    const doc = docs[step % docs.length];
    const tokens = [BOS, ...Array.from(doc).map((ch) => charIndex.get(ch)!), BOS];
    const n = Math.min(blockSize, tokens.length - 1);

    const keys: Value[][][] = Array.from({ length: nLayer }, () => []);
    const vals: Value[][][] = Array.from({ length: nLayer }, () => []);
    const losses: Value[] = [];
    for (let posId = 0; posId < n; posId++) {
      const tokenId = tokens[posId];
      const targetId = tokens[posId + 1];
      const logits = gpt(tokenId, posId, keys, vals);
      const probs = softmax(logits);
      losses.push(probs[targetId].log().neg());
    }
    const loss = sumValues(losses).mulS(1 / n);
    loss.backward();

    const lrT = learningRate * (1 - step / numSteps);
    const stepP1 = step + 1;
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * p.grad;
      vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * p.grad * p.grad;
      const mHat = mBuf[i] / (1 - beta1 ** stepP1);
      const vHat = vBuf[i] / (1 - beta2 ** stepP1);
      p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
      p.grad = 0;
    }

    process.stdout.write(`step ${String(step + 1).padStart(4)} / ${numSteps} | loss ${loss.data.toFixed(4)}\r`);
  }

  // --- 推理 ---
  const temperature = 0.5;
  console.log("\n--- inference (new, hallucinated names) ---");
  for (let sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
    const keys: Value[][][] = Array.from({ length: nLayer }, () => []);
    const vals: Value[][][] = Array.from({ length: nLayer }, () => []);
    let tokenId = BOS;
    const sample: string[] = [];
    for (let posId = 0; posId < blockSize; posId++) {
      const logits = gpt(tokenId, posId, keys, vals);
      const probs = softmax(logits.map((l) => l.mulS(1 / temperature)));
      // 加权随机采样
      const r = rng.random();
      let cum = 0;
      tokenId = 0;
      for (let i = 0; i < probs.length; i++) {
        cum += probs[i].data;
        if (r <= cum) { tokenId = i; break; }
      }
      if (tokenId === BOS) break;
      sample.push(uchars[tokenId]);
    }
    console.log(`sample ${String(sampleIdx + 1).padStart(2)}: ${sample.join("")}`);
  }
}

main();
