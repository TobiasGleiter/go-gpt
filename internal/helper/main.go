package helper

import (
	"fmt"
	"sort"
)

type Helper struct {
	Chars []rune
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
	h.Text := string(content)
}

func (h *Helper) CreateMappings() (map[rune]int, map[int]rune) {
	stoi := make(map[rune]int)
	itos := make(map[int]rune)
	for i, ch := range h.Chars {
		stoi[ch] = i
		itos[i] = ch
	}
	return stoi, itos
}

func (h *Helper) FindUniqueChars(text string) {
	chars := make(map[rune]bool)
	for _, char := range text {
		chars[char] = true
	}

    for char := range chars {
        h.Chars = append(h.Chars, char)
    }
    sort.Slice(h.Chars, func(i, j int) bool { return h.Chars[i] < h.Chars[j] })
}