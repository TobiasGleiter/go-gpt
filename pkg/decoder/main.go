package decoder

import (
	"strings"
)

type Decoder struct {}

func NewDecoder() *Decoder {
	return &Decoder{}
}

func (d *Decoder) Decode(l []int, itos map[int]rune) string {
	var decoded strings.Builder
	for _, i := range l {
		decoded.WriteRune(itos[i])
	}
	return decoded.String()
}