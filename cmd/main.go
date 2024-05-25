package main

import (
	"fmt"
	"math"

	h "github.com/TobiasGleiter/go-gpt/internal/helper"
	dc "github.com/TobiasGleiter/go-gpt/pkg/decoder"
	ec "github.com/TobiasGleiter/go-gpt/pkg/encoder"
	t "github.com/TobiasGleiter/go-gpt/pkg/tensor"
	bi "github.com/TobiasGleiter/go-gpt/pkg/neuralnetwork/bigram"
	"github.com/TobiasGleiter/go-gpt/pkg/neuralnetwork/optimizer"
)

func main() {
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


	optimizer := optimizer.NewSGD(parameters, 1e-5)

	batchSize = 32
	for epoch := 0; epoch < 1000; epoch++ {
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
}
