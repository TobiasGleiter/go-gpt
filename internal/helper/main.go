package helper

import (
	"sort"
	"io/ioutil"
	"log"
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