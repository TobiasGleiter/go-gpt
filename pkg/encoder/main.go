package encoder

type Encoder struct {}

func NewEncoder() *Encoder {
	return &Encoder{}
}

func (e *Encoder) Encode(s string, stoi map[rune]int) []int {
	var encoded []int
	for _, c := range s {
		encoded = append(encoded, stoi[c])
	}
	return encoded
}