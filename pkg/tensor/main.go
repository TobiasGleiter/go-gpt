package tensor

import (
	"fmt"
)

type Tensor struct {
	Shape []int
	Data  []int
}

func NewTensor(shape []int, data []int) *Tensor {
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