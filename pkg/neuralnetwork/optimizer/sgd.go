package optimizer

import (
	bi "github.com/TobiasGleiter/go-gpt/pkg/neuralnetwork/bigram"
)

type SGD struct {
	Parameters []*bi.Parameter
	Lr         float64
}

func NewSGD(parameters []*bi.Parameter, lr float64) *SGD {
	return &SGD{
		Parameters: parameters,
		Lr:         lr,
	}
}

func (opt *SGD) Step() {
	for _, param := range opt.Parameters {
		param.Value -= opt.Lr * param.Grad
	}
}

func (opt *SGD) ZeroGradients() {
	for _, param := range opt.Parameters {
		param.Grad = 0.0
	}
}
