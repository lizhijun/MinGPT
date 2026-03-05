// The most atomic way to train and run inference for a GPT in pure, dependency-free Go.
// This file is the complete algorithm.
// Everything else is just efficiency.
//
// Translated from @karpathy's microgpt.py

package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
)

// Let there be Autograd to recursively apply the chain rule through a computation graph
type Value struct {
	Data       float64
	Grad       float64
	children   []*Value
	localGrads []float64
}

func NewValue(data float64) *Value {
	return &Value{Data: data}
}

func NewValueWithChildren(data float64, children []*Value, localGrads []float64) *Value {
	return &Value{Data: data, children: children, localGrads: localGrads}
}

func (a *Value) Add(b *Value) *Value {
	return NewValueWithChildren(a.Data+b.Data, []*Value{a, b}, []float64{1, 1})
}

func (a *Value) AddScalar(s float64) *Value {
	return a.Add(NewValue(s))
}

func (a *Value) Mul(b *Value) *Value {
	return NewValueWithChildren(a.Data*b.Data, []*Value{a, b}, []float64{b.Data, a.Data})
}

func (a *Value) MulScalar(s float64) *Value {
	return a.Mul(NewValue(s))
}

func (a *Value) Pow(other float64) *Value {
	return NewValueWithChildren(math.Pow(a.Data, other), []*Value{a}, []float64{other * math.Pow(a.Data, other-1)})
}

func (a *Value) Log() *Value {
	return NewValueWithChildren(math.Log(a.Data), []*Value{a}, []float64{1.0 / a.Data})
}

func (a *Value) Exp() *Value {
	e := math.Exp(a.Data)
	return NewValueWithChildren(e, []*Value{a}, []float64{e})
}

func (a *Value) ReLU() *Value {
	d := 0.0
	g := 0.0
	if a.Data > 0 {
		d = a.Data
		g = 1.0
	}
	return NewValueWithChildren(d, []*Value{a}, []float64{g})
}

func (a *Value) Neg() *Value {
	return a.MulScalar(-1)
}

func (a *Value) Sub(b *Value) *Value {
	return a.Add(b.Neg())
}

func (a *Value) Div(b *Value) *Value {
	return a.Mul(b.Pow(-1))
}

func (a *Value) Backward() {
	// Topological sort
	topo := make([]*Value, 0)
	visited := make(map[*Value]bool)
	var buildTopo func(v *Value)
	buildTopo = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, child := range v.children {
			buildTopo(child)
		}
		topo = append(topo, v)
	}
	buildTopo(a)
	a.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		for j, child := range v.children {
			child.Grad += v.localGrads[j] * v.Grad
		}
	}
}

// sum of []*Value -> *Value
func sumValues(vals []*Value) *Value {
	s := NewValue(0)
	for _, v := range vals {
		s = s.Add(v)
	}
	return s
}

// Define the model architecture functions
func linear(x []*Value, w [][]*Value) []*Value {
	out := make([]*Value, len(w))
	for i, wo := range w {
		s := NewValue(0)
		for j, xi := range x {
			s = s.Add(wo[j].Mul(xi))
		}
		out[i] = s
	}
	return out
}

func softmax(logits []*Value) []*Value {
	maxVal := logits[0].Data
	for _, v := range logits[1:] {
		if v.Data > maxVal {
			maxVal = v.Data
		}
	}
	exps := make([]*Value, len(logits))
	for i, v := range logits {
		exps[i] = v.AddScalar(-maxVal).Exp()
	}
	total := sumValues(exps)
	out := make([]*Value, len(logits))
	for i, e := range exps {
		out[i] = e.Div(total)
	}
	return out
}

func rmsnorm(x []*Value) []*Value {
	n := len(x)
	s := NewValue(0)
	for _, xi := range x {
		s = s.Add(xi.Mul(xi))
	}
	ms := s.MulScalar(1.0 / float64(n))
	scale := ms.AddScalar(1e-5).Pow(-0.5)
	out := make([]*Value, n)
	for i, xi := range x {
		out[i] = xi.Mul(scale)
	}
	return out
}

func main() {
	// Let there be order among chaos
	rng := rand.New(rand.NewSource(42))

	// Let there be a Dataset
	if _, err := os.Stat("input.txt"); os.IsNotExist(err) {
		fmt.Println("Downloading input.txt...")
		resp, err := http.Get("https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt")
		if err != nil {
			panic(err)
		}
		defer resp.Body.Close()
		data, _ := io.ReadAll(resp.Body)
		os.WriteFile("input.txt", data, 0644)
	}
	fileData, err := os.ReadFile("input.txt")
	if err != nil {
		panic(err)
	}
	lines := strings.Split(string(fileData), "\n")
	docs := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			docs = append(docs, line)
		}
	}
	rng.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })
	fmt.Printf("num docs: %d\n", len(docs))

	// Let there be a Tokenizer
	charSet := make(map[rune]bool)
	for _, doc := range docs {
		for _, ch := range doc {
			charSet[ch] = true
		}
	}
	uchars := make([]rune, 0, len(charSet))
	for ch := range charSet {
		uchars = append(uchars, ch)
	}
	sort.Slice(uchars, func(i, j int) bool { return uchars[i] < uchars[j] })
	BOS := len(uchars)
	vocabSize := len(uchars) + 1
	fmt.Printf("vocab size: %d\n", vocabSize)

	// charIndex for fast lookup
	charIndex := make(map[rune]int)
	for i, ch := range uchars {
		charIndex[ch] = i
	}

	// Initialize the parameters
	nLayer := 1
	nEmbd := 16
	blockSize := 16
	nHead := 4
	headDim := nEmbd / nHead

	matrix := func(nout, nin int) [][]*Value {
		std := 0.08
		m := make([][]*Value, nout)
		for i := range m {
			m[i] = make([]*Value, nin)
			for j := range m[i] {
				m[i][j] = NewValue(rng.NormFloat64() * std)
			}
		}
		return m
	}

	stateDict := make(map[string][][]*Value)
	stateDict["wte"] = matrix(vocabSize, nEmbd)
	stateDict["wpe"] = matrix(blockSize, nEmbd)
	stateDict["lm_head"] = matrix(vocabSize, nEmbd)
	for i := 0; i < nLayer; i++ {
		stateDict[fmt.Sprintf("layer%d.attn_wq", i)] = matrix(nEmbd, nEmbd)
		stateDict[fmt.Sprintf("layer%d.attn_wk", i)] = matrix(nEmbd, nEmbd)
		stateDict[fmt.Sprintf("layer%d.attn_wv", i)] = matrix(nEmbd, nEmbd)
		stateDict[fmt.Sprintf("layer%d.attn_wo", i)] = matrix(nEmbd, nEmbd)
		stateDict[fmt.Sprintf("layer%d.mlp_fc1", i)] = matrix(4*nEmbd, nEmbd)
		stateDict[fmt.Sprintf("layer%d.mlp_fc2", i)] = matrix(nEmbd, 4*nEmbd)
	}

	// Flatten params — iterate in a deterministic order matching Python's dict insertion order
	paramKeys := []string{"wte", "wpe", "lm_head"}
	for i := 0; i < nLayer; i++ {
		paramKeys = append(paramKeys,
			fmt.Sprintf("layer%d.attn_wq", i),
			fmt.Sprintf("layer%d.attn_wk", i),
			fmt.Sprintf("layer%d.attn_wv", i),
			fmt.Sprintf("layer%d.attn_wo", i),
			fmt.Sprintf("layer%d.mlp_fc1", i),
			fmt.Sprintf("layer%d.mlp_fc2", i),
		)
	}
	params := make([]*Value, 0)
	for _, key := range paramKeys {
		mat := stateDict[key]
		for _, row := range mat {
			params = append(params, row...)
		}
	}
	fmt.Printf("num params: %d\n", len(params))

	// GPT forward function
	gpt := func(tokenID, posID int, keys, values [][][]*Value) []*Value {
		tokEmb := stateDict["wte"][tokenID]
		posEmb := stateDict["wpe"][posID]
		x := make([]*Value, nEmbd)
		for i := range x {
			x[i] = tokEmb[i].Add(posEmb[i])
		}
		x = rmsnorm(x)

		for li := 0; li < nLayer; li++ {
			// 1) Multi-head Attention block
			xResidual := x
			x = rmsnorm(x)
			q := linear(x, stateDict[fmt.Sprintf("layer%d.attn_wq", li)])
			k := linear(x, stateDict[fmt.Sprintf("layer%d.attn_wk", li)])
			v := linear(x, stateDict[fmt.Sprintf("layer%d.attn_wv", li)])
			keys[li] = append(keys[li], k)
			values[li] = append(values[li], v)
			xAttn := make([]*Value, 0, nEmbd)
			for h := 0; h < nHead; h++ {
				hs := h * headDim
				qH := q[hs : hs+headDim]
				kH := make([][]*Value, len(keys[li]))
				vH := make([][]*Value, len(values[li]))
				for t := range keys[li] {
					kH[t] = keys[li][t][hs : hs+headDim]
					vH[t] = values[li][t][hs : hs+headDim]
				}
				scale := 1.0 / math.Sqrt(float64(headDim))
				attnLogits := make([]*Value, len(kH))
				for t := range kH {
					s := NewValue(0)
					for j := 0; j < headDim; j++ {
						s = s.Add(qH[j].Mul(kH[t][j]))
					}
					attnLogits[t] = s.MulScalar(scale)
				}
				attnWeights := softmax(attnLogits)
				headOut := make([]*Value, headDim)
				for j := 0; j < headDim; j++ {
					s := NewValue(0)
					for t := range vH {
						s = s.Add(attnWeights[t].Mul(vH[t][j]))
					}
					headOut[j] = s
				}
				xAttn = append(xAttn, headOut...)
			}
			x = linear(xAttn, stateDict[fmt.Sprintf("layer%d.attn_wo", li)])
			for i := range x {
				x[i] = x[i].Add(xResidual[i])
			}
			// 2) MLP block
			xResidual = x
			x = rmsnorm(x)
			x = linear(x, stateDict[fmt.Sprintf("layer%d.mlp_fc1", li)])
			for i := range x {
				x[i] = x[i].ReLU()
			}
			x = linear(x, stateDict[fmt.Sprintf("layer%d.mlp_fc2", li)])
			for i := range x {
				x[i] = x[i].Add(xResidual[i])
			}
		}

		logits := linear(x, stateDict["lm_head"])
		return logits
	}

	// Let there be Adam, the blessed optimizer and its buffers
	learningRate := 0.01
	beta1 := 0.85
	beta2 := 0.99
	epsAdam := 1e-8
	mBuf := make([]float64, len(params))
	vBuf := make([]float64, len(params))

	// Repeat in sequence
	numSteps := 1000
	for step := 0; step < numSteps; step++ {
		// Take single document, tokenize it
		doc := docs[step%len(docs)]
		tokens := make([]int, 0, len(doc)+2)
		tokens = append(tokens, BOS)
		for _, ch := range doc {
			tokens = append(tokens, charIndex[ch])
		}
		tokens = append(tokens, BOS)
		n := blockSize
		if len(tokens)-1 < n {
			n = len(tokens) - 1
		}

		// Forward the token sequence through the model
		keys := make([][][]*Value, nLayer)
		vals := make([][][]*Value, nLayer)
		for i := range keys {
			keys[i] = make([][]*Value, 0)
			vals[i] = make([][]*Value, 0)
		}
		losses := make([]*Value, 0, n)
		for posID := 0; posID < n; posID++ {
			tokenID := tokens[posID]
			targetID := tokens[posID+1]
			logits := gpt(tokenID, posID, keys, vals)
			probs := softmax(logits)
			lossT := probs[targetID].Log().Neg()
			losses = append(losses, lossT)
		}
		loss := sumValues(losses).MulScalar(1.0 / float64(n))

		// Backward
		loss.Backward()

		// Adam optimizer update
		lrT := learningRate * (1 - float64(step)/float64(numSteps))
		stepP1 := float64(step + 1)
		for i, p := range params {
			mBuf[i] = beta1*mBuf[i] + (1-beta1)*p.Grad
			vBuf[i] = beta2*vBuf[i] + (1-beta2)*p.Grad*p.Grad
			mHat := mBuf[i] / (1 - math.Pow(beta1, stepP1))
			vHat := vBuf[i] / (1 - math.Pow(beta2, stepP1))
			p.Data -= lrT * mHat / (math.Sqrt(vHat) + epsAdam)
			p.Grad = 0
		}

		fmt.Printf("step %4d / %4d | loss %.4f\r", step+1, numSteps, loss.Data)
	}

	// Inference: may the model babble back to us
	temperature := 0.5
	fmt.Println("\n--- inference (new, hallucinated names) ---")
	for sampleIdx := 0; sampleIdx < 20; sampleIdx++ {
		keys := make([][][]*Value, nLayer)
		vals := make([][][]*Value, nLayer)
		for i := range keys {
			keys[i] = make([][]*Value, 0)
			vals[i] = make([][]*Value, 0)
		}
		tokenID := BOS
		sample := make([]rune, 0)
		for posID := 0; posID < blockSize; posID++ {
			logits := gpt(tokenID, posID, keys, vals)
			// Apply temperature
			scaledLogits := make([]*Value, len(logits))
			for i, l := range logits {
				scaledLogits[i] = l.MulScalar(1.0 / temperature)
			}
			probs := softmax(scaledLogits)
			// Weighted random choice
			r := rng.Float64()
			cumulative := 0.0
			tokenID = 0
			for i, p := range probs {
				cumulative += p.Data
				if r <= cumulative {
					tokenID = i
					break
				}
			}
			if tokenID == BOS {
				break
			}
			sample = append(sample, uchars[tokenID])
		}
		fmt.Printf("sample %2d: %s\n", sampleIdx+1, string(sample))
	}
}
