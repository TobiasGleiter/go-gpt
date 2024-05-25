package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"sort"
)

func main() {
	content, err := ioutil.ReadFile("../data/tinyshakespeare/input.txt")
	if err != nil {
		log.Fatal(err)
	}

	text := string(content)
	fmt.Println("Length of dataset in characters:", len(text))

	if len(text) > 0 { 
		fmt.Println(text[:50])
	}

	chars := make(map[rune]bool)
	for _, char := range text {
		chars[char] = true
	}

	var uniqueChars []rune
    for char := range chars {
        uniqueChars = append(uniqueChars, char)
    }
    sort.Slice(uniqueChars, func(i, j int) bool { return uniqueChars[i] < uniqueChars[j] })

    // Print unique characters and vocabulary size
    fmt.Println(string(uniqueChars))
    fmt.Println(len(uniqueChars))
}