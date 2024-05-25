package tensor

import (
	"fmt"
)

type Tensor struct {
	Data   [][]int
	Shape  [2]int
}

func NewTensor(shape [2]int, data [][]int) *Tensor {
	return &Tensor{
		Shape: shape,
		Data:  data,
	}
}

func (t *Tensor) Info() {
	fmt.Println("Shape:", t.Shape)
	fmt.Println("Data Type: int")
	fmt.Println("Data:", t.Data[:200])
}

// func getBatch(split string, data []int, blockSize, batchSize int) Tensor {
// 	x := make([][]int, batchSize)
// 	y := make([][]int, batchSize)

// 	for i := 0; i < batchSize; i++ {
// 		ix := rand.Intn(len(data) - blockSize)
// 		x[i] = data[ix : ix+blockSize]
// 		y[i] = data[ix+1 : ix+blockSize+1]
// 	}

// 	return Tensor{
// 		data:  append(x, y...),
// 		shape: [2]int{batchSize, blockSize},
// 	}
// }