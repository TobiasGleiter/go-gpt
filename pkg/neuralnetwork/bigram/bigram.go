package bigram

import (
	"math"
	"math/rand"
)

type BigramLanguageModel struct {
	tokenEmbeddingTable [][]float64
	vocabSize           int
}

func NewBigramLanguageModel(vocabSize int) *BigramLanguageModel {
	// Random weights in the language model
	// Table of the size vocabSize * vocabSize
	embedding := make([][]float64, vocabSize)
	for i := range embedding {
		embedding[i] = make([]float64, vocabSize)
		for j := range embedding[i] {
			embedding[i][j] = rand.Float64()
		}
	}
	return &BigramLanguageModel{
		tokenEmbeddingTable: embedding,
		vocabSize:           vocabSize,
	}
}

// idx gets the row with the given index in the language model (tokenEmbeddingTable)
func (m *BigramLanguageModel) Forward(idx []int, targets []int) ([][]float64, float64) {
	// Currently this does not use the T (Time)
	B := len(idx)
	logits := make([][]float64, B) // (B, T, C) e.g. BatchSize (4), Time (8), Channel (65 voc size) 
	for i := range logits {
		logits[i] = make([]float64, m.vocabSize) 
		for j := range logits[i] {
			logits[i][j] = m.tokenEmbeddingTable[idx[i]][j] 
		}
	}

	// Calculate loss (how good the prediction is)
	var loss float64
	if targets != nil {
		for i := 0; i < B; i++ {
			for j := 0; j < m.vocabSize; j++ {
				if j == targets[i] {
					loss -= math.Log(logits[i][j])
				}
			}
		}
		loss /= float64(B)
	}

	return logits, loss
}

func (m *BigramLanguageModel) Generate(idx []int, maxNewTokens int) []int {
	for i := 0; i < maxNewTokens; i++ {
		logits, _ := m.Forward(idx, nil)
		lastLogits := logits[len(logits)-1]
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