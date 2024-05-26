package main

import (
	"fmt"
	"math"
	"math/rand"

	h "github.com/TobiasGleiter/go-gpt/internal/helper"
	dc "github.com/TobiasGleiter/go-gpt/pkg/decoder"
	ec "github.com/TobiasGleiter/go-gpt/pkg/encoder"
	t "github.com/TobiasGleiter/go-gpt/pkg/tensor"
	bi "github.com/TobiasGleiter/go-gpt/pkg/neuralnetwork/bigram"
	"github.com/TobiasGleiter/go-gpt/pkg/neuralnetwork/optimizer"
)

func main() {
	rand.Seed(1337)

	helper := h.NewHelper()
	
	helper.LoadTextFile("../data/tinyshakespeare/input.txt")
	helper.FindUniqueChars()
	stoi, itos := helper.CreateMappings()

	encoder := ec.NewEncoder()
	decoder := dc.NewDecoder()

	testStr := "hii there"
	encoded := encoder.Encode(testStr, stoi) // encoder: take a string, output a list of integers
	fmt.Println(encoded)
	decoded := decoder.Decode(encoded, itos) // decoder: take a list of integers, output a string
	fmt.Println(decoded)


	fullTextEncoded := encoder.Encode(helper.Text, stoi)
	
	n := int(0.9 * float64(len(fullTextEncoded)))
	trainingsData := fullTextEncoded[:n] // first 90 % will be train, rest validation
	//validationData := fullTextEncoded[n:]

	
	batchSize := 4 // how many independent sequences will we process in parallel
	blockSize := 8 // or also known as context-length
	
	//helper.ShowSimpleTokenPrediction(trainingsData, blockSize) // Show simple token prediction (token = number)

	tensor := t.NewTensor()

	xbTensor, ybTensor := tensor.GetBatch(trainingsData, blockSize, batchSize)
	xb := tensor.TensorToSlice(xbTensor)
	yb := tensor.TensorToSlice(ybTensor)

	// helper.ShowBatches(xbTensor, ybTensor, batchSize) // Shows the batches for parallel processing where xb predicts yb
	// helper.ShowBatchesTokenPrediction(xbTensor, ybTensor, batchSize, blockSize)

	fmt.Println(xbTensor.Data)
	fmt.Println(ybTensor.Data)

	vocSize := len(helper.UniqueChars)
	model := bi.NewBigramLanguageModel(vocSize)


	_, loss := model.Forward(xb, yb)
	// fmt.Println("Logits:", logits) // Logits represent ???
	fmt.Println("Loss:", loss) // Higher number should be better?

	generated := model.Generate([]int{0}, 100)
	fmt.Println("Generated sequence:", decoder.Decode(generated, itos)) // This generates garbage :D

	var parameters []*bi.Parameter
	for _, row := range model.TokenEmbeddingTable {
		for _, param := range row {
			parameters = append(parameters, param)
		}
	}

	fmt.Println("Parameters:", len(parameters))


	optimizer := optimizer.NewSGD(parameters, 1e-3)

	batchSize = 32
	for epoch := 0; epoch < 100; epoch++ {
		xbTensor, ybTensor := tensor.GetBatch(trainingsData, blockSize, batchSize)
		xb := tensor.TensorToSlice(xbTensor)
		yb := tensor.TensorToSlice(ybTensor)

		logits, loss := model.Forward(xb, yb)

		if math.IsNaN(loss) {
			fmt.Printf("NaN encountered at epoch %d\n", epoch)
			break
		}
		fmt.Printf("Epoch %d, Loss: %f\n", epoch, loss)

		model.Backward(xb, yb, logits)
		optimizer.Step()
		optimizer.ZeroGradients()
	}

	generated = model.Generate([]int{0}, 100)
	fmt.Println("Generated sequence:", decoder.Decode(generated, itos)) // This generates garbage :D


	// Version 1: Averaging past context with for loop, the weakes form of...
	B := 2 // Number of batches
	T := 4 // Sequence length
	C := 3 // Embedding dimension

	// Example input tensor x of shape (B, T, C)
	x := [][][]float64{
		{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
			{10.0, 11.0, 12.0},
		},
		{
			{13.0, 14.0, 15.0},
			{16.0, 17.0, 18.0},
			{19.0, 20.0, 21.0},
			{22.0, 23.0, 24.0},
		},
	}

	// Initialize the output tensor xbow with shape (B, T, C)
	xbow := make([][][]float64, B)
	for b := range xbow {
		xbow[b] = make([][]float64, T)
		for t := range xbow[b] {
			xbow[b][t] = make([]float64, C)
		}
	}

	// Compute the mean of all previous tokens up to the current token
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			sum := make([]float64, C)
			count := float64(t + 1) // Number of elements to include in the mean
			for i := 0; i <= t; i++ {
				for j := 0; j < C; j++ {
					sum[j] += x[b][i][j]
				}
			}
			for j := 0; j < C; j++ {
				xbow[b][t][j] = sum[j] / count
			}
		}
	}

	fmt.Println(x[0])
	fmt.Println(xbow[0])

	// Print the output tensor xbow
	for b := range xbow {
		fmt.Printf("Batch %d:\n", b)
		for t := range xbow[b] {
			fmt.Printf("Timestep %d: %v\n", t, xbow[b][t])
		}
	}
}
