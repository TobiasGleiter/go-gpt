package main

import (
	"fmt"

	h "github.com/TobiasGleiter/go-gpt/internal/helper"
	dc "github.com/TobiasGleiter/go-gpt/pkg/decoder"
	ec "github.com/TobiasGleiter/go-gpt/pkg/encoder"
	t "github.com/TobiasGleiter/go-gpt/pkg/tensor"
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

	blockSize := 8 // or context-length

	// Prediction of the next "token"
	// The transformer sees all combinations from one to blockSize (more than blockSize could not be predicted)
	x := trainingsData[:blockSize]
	y := trainingsData[1:blockSize+1] // 8 individual numbers, thats why blocksize+1 (18, 45 => 56)
	for t := 0; t < blockSize; t++ {
		context := x[:t+1]
		target := y[t]
		fmt.Printf("When input is %v the target: %d\n", context, target)
	}

	batchSize := 4 // how many independent sequences will we process in parallel
	blockSize = 8
	
	tensor := t.NewTensor()

	xb := tensor.GetBatch(trainingsData, blockSize, batchSize)
	yb := tensor.GetBatch(trainingsData, blockSize, batchSize)

	fmt.Println("inputs:")
	fmt.Println(xb.Shape)
	for _, row := range xb.Data[:batchSize] {
		fmt.Println(row)
	}

	fmt.Println("targets:")
	fmt.Println(yb.Shape)
	for _, row := range yb.Data[:batchSize] {
		fmt.Println(row)
	}

	fmt.Println("----")

	for b := 0; b < batchSize; b++ { // batch dimension
		for t := 0; t < blockSize; t++ { // time dimension
			context := xb.Data[b][:t+1]
			target := yb.Data[b][t]
			fmt.Printf("when input is %v the target: %d\n", context, target)
		}
	}

	print(xb)
}
