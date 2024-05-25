package tensor

import (
	"math/rand"
	"fmt"
)

type Tensor struct {
	Data   [][]int
	Shape  [2]int
}

func NewTensor() *Tensor {
	return &Tensor{}
}

func (t *Tensor) Info() {
	fmt.Println("Shape:", t.Shape)
	fmt.Println("Data Type: int")
	fmt.Println("Data:", t.Data[:200])
}

func (t *Tensor) GetBatch(data []int, blockSize, batchSize int) Tensor {
	x := make([][]int, batchSize)
	y := make([][]int, batchSize)

	for i := 0; i < batchSize; i++ {
		ix := rand.Intn(len(data) - blockSize)
		x[i] = data[ix : ix+blockSize]
		y[i] = data[ix+1 : ix+blockSize+1]
	}

	return Tensor{
		Data:  append(x, y...),
		Shape: [2]int{batchSize, blockSize},
	}
}