package helper

import (
	"fmt"
	"sort"
	"io/ioutil"
	"log"

	t "github.com/TobiasGleiter/go-gpt/pkg/tensor"
)

type Helper struct {
	UniqueChars []rune
	Text string
}

func NewHelper() *Helper {
	return &Helper{}
}

func (h *Helper) LoadTextFile(path string) {
	content, err := ioutil.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	h.Text = string(content)
}

func (h *Helper) FindUniqueChars() {
	chars := make(map[rune]bool)
	for _, char := range h.Text {
		chars[char] = true
	}

    for char := range chars {
        h.UniqueChars = append(h.UniqueChars, char)
    }
    sort.Slice(h.UniqueChars, func(i, j int) bool { return h.UniqueChars[i] < h.UniqueChars[j] })
}

func (h *Helper) CreateMappings() (map[rune]int, map[int]rune) {
	stoi := make(map[rune]int)
	itos := make(map[int]rune)
	for i, ch := range h.UniqueChars {
		stoi[ch] = i
		itos[i] = ch
	}
	return stoi, itos
}

func (h *Helper) ShowSimpleTokenPrediction(data []int, blockSize int) {
	// Prediction of the next "token"
	// The transformer sees all combinations from one to blockSize (more than blockSize could not be predicted)
	x := data[:blockSize]
	y := data[1:blockSize+1] // blockSize individual numbers can predict blocksize+1 numbers (e.g. 1 => 42, 1 42 => 30, ...)
	for t := 0; t < blockSize; t++ {
		context := x[:t+1]
		target := y[t]
		fmt.Printf("When input is %v the target: %d\n", context, target)
	}
}

func (h *Helper) ShowBatches(xb, yb t.Tensor, batchSize int) {
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
}

func (h *Helper) ShowBatchesTokenPrediction(xb, yb t.Tensor, batchSize, blockSize int) {
	for b := 0; b < batchSize; b++ { // batch dimension
		for t := 0; t < blockSize; t++ { // time dimension
			context := xb.Data[b][:t+1]
			target := yb.Data[b][t]
			fmt.Printf("when input is %v the target: %d\n", context, target)
		}
	}
}