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
	dataTensor := t.NewTensor([]int{len(fullTextEncoded)}, fullTextEncoded)

	dataTensor.Info()

}