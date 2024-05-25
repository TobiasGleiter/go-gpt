package bigram

import (
	"math"
	"math/rand"
)

type Parameter struct {
	Value   float64
	Grad    float64
}

type BigramLanguageModel struct {
	TokenEmbeddingTable [][]*Parameter
	VocabSize           int
}

func NewBigramLanguageModel(vocabSize int) *BigramLanguageModel {
	table := make([][]*Parameter, vocabSize)
	for i := range table {
		table[i] = make([]*Parameter, vocabSize)
		for j := range table[i] {
			table[i][j] = &Parameter{Value: rand.Float64()}
		}
	}
	return &BigramLanguageModel{
		TokenEmbeddingTable: table,
		VocabSize:           vocabSize,
	}
}

// idx gets the row with the given index in the language model (tokenEmbeddingTable)
func (m *BigramLanguageModel) Forward(idx [][]int, targets [][]int) ([][][]float64, float64) {
	B := len(idx) // Batch size
	T := len(idx[0]) // Sequence length (time dimension)
	logits := make([][][]float64, B) // (B, T, C)
	for i := range logits {
		logits[i] = make([][]float64, T)
		for t := range logits[i] {
			logits[i][t] = make([]float64, m.VocabSize)
			for j := range logits[i][t] {
				logits[i][t][j] = m.TokenEmbeddingTable[idx[i][t]][j].Value
			}
		}
	}

	// Calculate Cross-Entropy Loss
	var loss float64
	if targets != nil {
		for i := 0; i < B; i++ {
			for t := 0; t < T; t++ {
				target := targets[i][t]
				loss -= math.Log(logits[i][t][target])
			}
		}
		loss /= float64(B * T)
	}

	return logits, loss
}

func (m *BigramLanguageModel) Backward(idx [][]int, targets [][]int, logits [][][]float64) {
	B := len(idx) // Batch size
	T := len(idx[0]) // Sequence length (time dimension)
	for i := 0; i < B; i++ {
		for t := 0; t < T; t++ {
			target := targets[i][t]
			for j := 0; j < m.VocabSize; j++ {
				grad := logits[i][t][j]
				if j == target {
					grad -= 1
				}
				m.TokenEmbeddingTable[idx[i][t]][j].Grad += grad
			}
		}
	}
}

func (m *BigramLanguageModel) Generate(idx []int, maxNewTokens int) []int {
	for i := 0; i < maxNewTokens; i++ {
		batch := [][]int{idx}
		logits, _ := m.Forward(batch, nil) // Don't use loss
		lastLogits := logits[0][len(idx)-1]
		nextIdx := sampleFromDistribution(lastLogits)
		idx = append(idx, nextIdx)
	}
	return idx
}



func sampleFromDistribution(probs []float64) int {
	total := 0.0
	for _, prob := range probs {
		total += prob
	}
	rnd := rand.Float64() * total
	for i, prob := range probs {
		rnd -= prob
		if rnd <= 0 {
			return i
		}
	}
	return len(probs) - 1
}