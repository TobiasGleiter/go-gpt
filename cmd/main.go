package main

import (
	"fmt"
	"math/rand"

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
	// dataTensor := t.NewTensor([]int{len(fullTextEncoded), 1}, fullTextEncoded)

	// dataTensor.Info()

	
	n := int(0.9 * float64(len(fullTextEncoded)))
	trainingData := fullTextEncoded[:n] // first 90 % will be train, rest validation
	//validationData := fullTextEncoded[n:]

	blockSize := 8 // or context-length

	// Prediction of the next "token"
	// The transformer sees all combinations from one to blockSize (more than blockSize could not be predicted)
	x := trainingData[:blockSize]
	y := trainingData[1:blockSize+1] // 8 individual numbers, thats why blocksize+1
	for t := 0; t < blockSize; t++ {
		context := x[:t+1]
		target := y[t]
		fmt.Printf("When input is %v the target: %d\n", context, target)
	}

	batchSize := 4
	blockSize = 8
	
	xb := getBatch("train", trainingData, blockSize, batchSize)
	yb := getBatch("train", trainingData, blockSize, batchSize)

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
}

func getBatch(split string, data []int, blockSize, batchSize int) t.Tensor {
	x := make([][]int, batchSize)
	y := make([][]int, batchSize)

	for i := 0; i < batchSize; i++ {
		ix := rand.Intn(len(data) - blockSize)
		x[i] = data[ix : ix+blockSize]
		y[i] = data[ix+1 : ix+blockSize+1]
	}

	return t.Tensor{
		Data:  append(x, y...),
		Shape: [2]int{batchSize, blockSize},
	}
}
