package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"

	"github.com/koeng101/dnadesign/lib/bio"
)

// TokenizeProtein tokenizes protein sequences into uint8.
func TokenizeProtein(sequence string) ([]uint8, error) {
	// Switch statements are faster than maps
	// https://adayinthelifeof.nl/2021/03/04/go-map-vs-switch.html
	// https://www.reddit.com/r/golang/comments/lxju7f/benchmarking_maps_vs_switches/
	tokens := make([]uint8, len(sequence)+1) // +1 for end token, which is the default 0
	var token uint8

	// Tokens: end_token, "ACDEFGHIKLMNPQRSTVWYUO*BXZ"
	// {"A":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"K":9,"L":10,"M":11,"N":12,"P":13,"Q":14,"R":15,"S":16,"T":17,"V":18,"W":19,"Y":20,"U":21,"O":22,"*":23,"B":24,"X":25,"Z":26}
	for i, aminoAcid := range sequence {
		switch aminoAcid {
		case 'A':
			token = 1
		case 'C':
			token = 2
		case 'D':
			token = 3
		case 'E':
			token = 4
		case 'F':
			token = 5
		case 'G':
			token = 6
		case 'H':
			token = 7
		case 'I':
			token = 8
		case 'K':
			token = 9
		case 'L':
			token = 10
		case 'M':
			token = 11
		case 'N':
			token = 12
		case 'P':
			token = 13
		case 'Q':
			token = 14
		case 'R':
			token = 15
		case 'S':
			token = 16
		case 'T':
			token = 17
		case 'V':
			token = 18
		case 'W':
			token = 19
		case 'Y':
			token = 20
		case 'U': // Selenocysteine
			token = 21
		case 'O': // Pyrrolysine
			token = 22
		case '*': // Stop codon
			token = 23
		case 'B': // Aspartic acid or Asparagine
			token = 24
		case 'X': // Any amino acid
			token = 25
		case 'Z': // Glutamic acid or Glutamine
			token = 26
		default:
			return tokens, fmt.Errorf("Got unknown amino acid. Must be in list of ACDEFGHIKLMNPQRSTVWYUO*BXZ. Got: %c", aminoAcid)
		}
		tokens[i] = token
	}
	return tokens, nil
}

func main() {
	/*
		Open buffered writers, write header
	*/
	// Create training file
	trainFileRaw, err := os.Create("train.bin")
	if err != nil {
		log.Fatalf("Failed to open trainFile: %s", err)
	}
	defer trainFileRaw.Close()
	trainFile := bufio.NewWriter(trainFileRaw)
	var trainFileTokens int32

	// Create validation file
	valFileRaw, err := os.Create("val.bin")
	if err != nil {
		log.Fatalf("Failed to open valFile: %s", err)
	}
	defer valFileRaw.Close()
	valFile := bufio.NewWriter(valFileRaw)
	var valFileTokens int32

	// Write headers to both files
	// We write the header here, as defined in Karpathy's llm.c
	header := make([]int32, 256) // Create a slice for 256 int32
	header[0] = 20240520         // Set magic number
	header[1] = 1                // Set version info
	header[2] = 0                // Set the third int to zero, for now

	// Convert the header to bytes and write it.
	for _, file := range []*bufio.Writer{valFile, trainFile} {
		for _, value := range header {
			err := binary.Write(file, binary.LittleEndian, value)
			if err != nil {
				log.Fatalf("Got error writing header: %s", err)
			}
		}
	}

	/*
		Randomize validation set
	*/
	totalProteinSequences := 190_000_000
	numInValidation := 1_000_000
	var seed int64 = 42 // time.Now().Unix()
	rand.Seed(seed)

	// Create a map to track which numbers have already been chosen
	chosen := make(map[int]bool)

	// Generate X unique random numbers
	result := make([]int, 0, numInValidation)
	for len(result) < numInValidation {
		num := rand.Intn(totalProteinSequences)
		if !chosen[num] {
			chosen[num] = true
			result = append(result, num)
		}
	}

	/*
		Start of tokenization
	*/
	parser := bio.NewFastaParser(os.Stdin)
	var i int
	for {
		if (i % 50_000) == 0 {
			log.Printf("Processed: %d\n", i)
		}
		protein, err := parser.Next()
		if err != nil {
			break
		}
		sequence := strings.ToUpper(protein.Sequence)
		tokens, err := TokenizeProtein(sequence)
		if err != nil {
			log.Fatalf("Failed to tokenize protein: %s", err)
		}
		// Set into training or validation
		if chosen[i] {
			for _, token := range tokens {
				_, _ = valFile.Write([]byte{byte(token), 0})
			}
			valFileTokens += int32(len(tokens))
		} else {
			for _, token := range tokens {
				_, _ = trainFile.Write([]byte{byte(token), 0})
			}
			trainFileTokens += int32(len(tokens))
		}
		i++
	}

	/*
		Complete writing of files
	*/
	trainFile.Flush()
	valFile.Flush()

	// Calculate the offset for the 3rd int32
	// int32 is 4 bytes, so the 3rd int32 starts at byte 8 (0-based indexing)
	offset := int64(2 * 4)
	trainAndValTotals := []int32{valFileTokens, trainFileTokens}
	for i, file := range []*os.File{valFileRaw, trainFileRaw} {
		// Seek to the position of the 3rd int32
		_, err = file.Seek(offset, 0)
		if err != nil {
			log.Fatalf("Failed to seek in file: %v", err)
		}
		// Write zero value for int32 at this position
		err = binary.Write(file, binary.LittleEndian, trainAndValTotals[i])
		if err != nil {
			log.Fatalf("Failed to write to file: %v", err)
		}

	}
}
