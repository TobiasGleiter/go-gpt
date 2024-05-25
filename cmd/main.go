package main

import (
	"fmt"

	h "github.com/TobiasGleiter/go-gpt/internal/helper"
	dc "github.com/TobiasGleiter/go-gpt/pkg/decoder"
	ec "github.com/TobiasGleiter/go-gpt/pkg/encoder"
	t "github.com/TobiasGleiter/go-gpt/pkg/tensor"
	bi "github.com/TobiasGleiter/go-gpt/pkg/neuralnetwork/bigram"
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

	xb := tensor.GetBatch(trainingsData, blockSize, batchSize)
	yb := tensor.GetBatch(trainingsData, blockSize, batchSize)

	//helper.ShowBatches(xb, yb, batchSize) // Shows the batches for parallel processing where xb predicts yb
	//helper.ShowBatchesTokenPrediction(xb, yb, batchSize, blockSize)

	fmt.Println(xb.Data) // input to the transformer
	fmt.Println(yb.Data)

	vocSize := len(helper.UniqueChars)
	model := bi.NewBigramLanguageModel(vocSize)

	// logits, loss := model.Forward(xb.Data[0], yb.Data[0])
	// fmt.Println("Logits:", logits) // Logits represent ???
	// fmt.Println("Loss:", loss) // Higher number should be better?

	generated := model.Generate([]int{0}, 100)
	fmt.Println("Generated sequence:", decoder.Decode(generated, itos))
}
