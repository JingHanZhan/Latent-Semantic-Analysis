package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

//HASHSIZE is hashtable size
//const HASHSIZE = 100000

var (
	i             = 0
	collision     = 0
	docNumber     = 0
	wordDimension = 0
)

const rank = 4

func main() {
	checkParameter()
	clearOutputFile()
	start := time.Now()
	var (
		nodeTable = make([]*node, *args.h) //[*args.m]*node
		keyTable  []*key
	)
	//fmt.Println("cap(nodeTable):", unsafe.Sizeof(node{"1", 0, 0, 0, nil}))
	creatBasicTable(nodeTable[:], keyTable)
	fmt.Println("How many collision:", collision)
	clearTableCount(nodeTable[:])
	doc2vec(nodeTable[:], keyTable)

	/* 下面兩條應該包進去doc2vec function內 */
	// outputNodeTable(nodeTable[:])
	// outputVector(nodeTable[:])

	//fmt.Println("test nodetable:", nodeTable[0].count)
	fmt.Println("time:", time.Now().Sub(start).Seconds())
	//test()
	fmt.Println("wordDimension:", wordDimension)
	fmt.Println("docNumber:", docNumber)

	/* compute svd  */
	docvec := loadMatrixData()
	A := mat.NewDense(20, 95, docvec)
	docSVDMatrix(A, rank)

}

/* load matrix from file like txt */
func loadMatrixData() []float64 {
	var docvec []float64
	fmt.Println("load matrix data ...")
	vecFile, err := os.Open("doc2vec.txt")
	if err != nil {
		fmt.Println("Open file fail !")
	}
	defer vecFile.Close()
	scanner := bufio.NewReader(vecFile)
	stringBuf, readStringErr := scanOneWord(scanner)
	for readStringErr == nil {
		if strings.Contains(stringBuf, "\n") {
			s := strings.Split(stringBuf, "\n")
			stringBuf = s[0]

			// 去除空格
			stringBuf = strings.Replace(stringBuf, " ", "", -1)
			// 去除换行符
			stringBuf = strings.Replace(stringBuf, "\n", "", -1)
			if string2float, err := strconv.ParseFloat(stringBuf, 32); err == nil {
				docvec = append(docvec, string2float)
			}
			stringBuf = s[1]
		} else {
			// 去除空格
			stringBuf = strings.Replace(stringBuf, " ", "", -1)
			if string2float, err := strconv.ParseFloat(stringBuf, 32); err == nil {
				docvec = append(docvec, string2float)
			}
			stringBuf, readStringErr = scanOneWord(scanner)
		}
	}
	// fmt.Println(docvec)
	return docvec
}

func docSVDMatrix(A *mat.Dense, rank int) {
	var (
		svd mat.SVD
		um  mat.Dense
	)

	svd.Factorize(A, mat.SVDThin)
	svd.UTo(&um)
	s := svd.Values(nil)
	dim1, _ := um.Dims()

	if rank > len(s) {
		fmt.Println("SVD rand set error")
		return
	}

	result := make([]float64, dim1*rank)
	//fmt.Println(dim1, dim2)
	for i := 0; i < dim1*rank; i++ {
		result[i] = s[i%rank] * um.At(i/rank, i%rank)
	}
	B := mat.NewDense(dim1, rank, result)
	matPrint(B)
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v \n", fa)
}

func test() {
	//test2
	hashFile, err := os.Open("test2.txt")
	if err != nil {
		fmt.Println("Open file fail !")
	}
	defer hashFile.Close()
	scanner := bufio.NewReader(hashFile)
	var stringBuf string
	for i := 0; i < 3; i++ {
		stringBuf, _ = scanOneLine(scanner)
	}
	hashNumber := hash(&stringBuf)
	fmt.Println("test function => hashNumber:", hashNumber)

	//test3
	hashFile2, err2 := os.Open("test3.txt")
	if err2 != nil {
		fmt.Println("Open file fail !")
	}
	defer hashFile2.Close()
	scanner2 := bufio.NewReader(hashFile2)
	var stringBuf2 string
	for j := 0; j < 3; j++ {
		stringBuf2, _ = scanOneWord(scanner2)
	}
	stringBuf2 = stringBuf2 + "\r\n"
	hashNumber2 := hash(&stringBuf2)
	fmt.Println("test function => hashNumber2:", hashNumber2)
	fmt.Println(stringBuf)
	fmt.Println(stringBuf2)
}

var args struct {
	m *uint64
	s *uint64
	h *uint64
}

type node struct {
	//keypos uint
	key   string
	count int
	idf   float64
	tfidf float64
	next  *node
}

type key struct {
	keypos *string
	count  *int
	idf    *float64
	tfidf  *float64
}

func doc2vec(nodeTable []*node, keyTable []*key) {
	fmt.Println("Start doc to vector ...")
	hashFile, err := os.Open("test.txt")
	if err != nil {
		fmt.Println("Open file fail !")
	}
	defer hashFile.Close()
	scanner := bufio.NewReader(hashFile)
	stringBuf, readStringErr := scanOneWord(scanner)
	//stringBuf = stringBuf + "\n"

	for readStringErr == nil {
		if strings.Contains(stringBuf, "\n") {
			s := strings.Split(stringBuf, "\r\n")
			// fmt.Println(s)
			//fmt.Println("contain change line:", stringBuf)
			stringBuf = s[0]

			// 去除空格
			stringBuf = strings.Replace(stringBuf, " ", "", -1)
			// 去除换行符
			stringBuf = strings.Replace(stringBuf, "\n", "", -1)

			if uint64(len(stringBuf)) <= (*args.s + 2) {
				hashNumber := hash(&stringBuf)
				nodeTable[hashNumber], keyTable = addVecWord(nodeTable[hashNumber], stringBuf, keyTable)
			}
			computeTFIDF(nodeTable[:])
			outputNodeTable(nodeTable[:])
			outputVector(nodeTable[:])
			clearTableCount(nodeTable[:])
			//stringBuf, readStringErr = scanOneWord(scanner)
			stringBuf = s[1]
		} else {
			// 去除空格
			stringBuf = strings.Replace(stringBuf, " ", "", -1)
			if uint64(len(stringBuf)) <= (*args.s + 2) {
				hashNumber := hash(&stringBuf)
				nodeTable[hashNumber], keyTable = addVecWord(nodeTable[hashNumber], stringBuf, keyTable)
			}
			stringBuf, readStringErr = scanOneWord(scanner)
		}
	}
}

/* 原本的BasicTable funcion */
// func creatBasicTable(nodeTable []*node, keyTable []*key) {
// 	hashFile, err := os.Open(os.Args[1])
// 	if err != nil {
// 		fmt.Println("Open file fail !")
// 	}
// 	defer hashFile.Close()
// 	scanner := bufio.NewReader(hashFile)
// 	stringBuf, readStringErr := scanOneLine(scanner)
// 	for readStringErr == nil {
// 		if uint64(len(stringBuf)) <= (*args.s + 2) {
// 			hashNumber := hash(&stringBuf)
// 			nodeTable[hashNumber], keyTable = addWord(nodeTable[hashNumber], stringBuf, keyTable)
// 		}
// 		stringBuf, readStringErr = scanOneLine(scanner)
// 	}
// }

/* 改版BasicTable，施工中 */
func creatBasicTable(nodeTable []*node, keyTable []*key) {
	fmt.Println("Creat basic table ...")
	hashFile, err := os.Open(os.Args[1])
	if err != nil {
		fmt.Println("Open file fail !")
	}
	defer hashFile.Close()
	scanner := bufio.NewReader(hashFile)
	stringBuf, readStringErr := scanOneWord(scanner)

	// // 去除空格
	// stringBuf = strings.Replace(stringBuf, " ", "", -1)
	// // 去除换行符
	// stringBuf = strings.Replace(stringBuf, "\n", "", -1)

	for readStringErr == nil {
		if strings.Contains(stringBuf, "\n") {
			s := strings.Split(stringBuf, "\n")
			stringBuf = s[0]

			// 去除空格
			stringBuf = strings.Replace(stringBuf, " ", "", -1)
			// 去除换行符
			stringBuf = strings.Replace(stringBuf, "\n", "", -1)

			if uint64(len(stringBuf)) <= (*args.s + 2) {
				hashNumber := hash(&stringBuf)
				nodeTable[hashNumber], keyTable = addWord(nodeTable[hashNumber], stringBuf, keyTable)
			}
			stringBuf = s[1]
			idfDenominator(nodeTable[:])
			clearTableCount(nodeTable[:])
			docNumber++
		} else {
			// 去除空格
			stringBuf = strings.Replace(stringBuf, " ", "", -1)
			if uint64(len(stringBuf)) <= (*args.s + 2) {
				hashNumber := hash(&stringBuf)
				nodeTable[hashNumber], keyTable = addWord(nodeTable[hashNumber], stringBuf, keyTable)
			}
			stringBuf, readStringErr = scanOneWord(scanner)
		}
	}
	computeIDF(nodeTable[:])
}

func clearTableCount(nodeTable []*node) {
	for n := 0; n < len(nodeTable); n++ {
		for p := nodeTable[n]; p != nil; p = p.next {
			p.count = 0
			//nodeTableFile.WriteString(fmt.Sprintf("%d %s", p.count, p.key))
		}
	}
}

/* 統計idf的分母，也就是每個詞被多少文見提及 */
func idfDenominator(nodeTable []*node) {
	for n := 0; n < len(nodeTable); n++ {
		for p := nodeTable[n]; p != nil; p = p.next {
			if p.count != 0 {
				p.idf++
			}
		}
	}
}

/* 將docNumber 除以 idfDenominator 取 log，計算出idf數值*/
func computeIDF(nodeTable []*node) {
	for n := 1; n < len(nodeTable); n++ {
		for p := nodeTable[n]; p != nil; p = p.next {
			p.idf = math.Log10(float64(docNumber) / p.idf)
			wordDimension++
		}
	}
}

/* 計算TFIDF數值 */
func computeTFIDF(nodeTable []*node) {
	for n := 0; n < len(nodeTable); n++ {
		for p := nodeTable[n]; p != nil; p = p.next {
			p.tfidf = (float64(p.count) / float64(wordDimension)) * p.idf
		}
	}
}

func checkParameter() {
	for i := 1; i < len(os.Args); i++ {
		switch os.Args[i] {
		case "-m":
			parmBuf, err := strconv.ParseUint(os.Args[i+1], 10, 64)
			if err != nil {
				fmt.Println("Parameter m error: ", os.Args[i+1])
			} else {
				args.m = &parmBuf
				fmt.Println("使用參數:m = ", *args.m)
			}
		case "-s":
			parmBuf, err := strconv.ParseUint(os.Args[i+1], 10, 64)
			if err != nil {
				fmt.Println("Parameter s error: ", os.Args[i+1])
			} else {
				args.s = &parmBuf
				fmt.Println("使用參數:s = ", *args.s)
			}
		case "-h":
			parmBuf, err := strconv.ParseUint(os.Args[i+1], 10, 64)
			if err != nil {
				fmt.Println("Parameter h error: ", os.Args[i+1])
			} else {
				args.h = &parmBuf
				fmt.Println("使用參數:h = ", *args.h)
			}
		}
	}
}

/* 計算單字詞頻，建立basic node table，遇到新的單字擴增*/
func addWord(newWord *node, stringBuf string, keyTable []*key) (*node, []*key) {
	memoryLimit()
	i++
	fmt.Printf("Add word : %d\r", i)
	if newWord == nil {
		newWord = &node{stringBuf, 1, 0, 0, nil}
		keyTable = append(keyTable, &key{&newWord.key, &newWord.count, &newWord.idf, &newWord.tfidf})
		return newWord, keyTable
	}

	for w := newWord; w != nil; w = w.next {
		if w.key == stringBuf {
			w.count++
			return newWord, keyTable
		}
		if w.next == nil {
			collision++
			newNode := &node{stringBuf, 1, 0, 0, nil}
			keyTable = append(keyTable, &key{&newNode.key, &newNode.count, &newNode.idf, &newNode.tfidf})
			w.next = newNode
			return newWord, keyTable
		}
	}
	return newWord, keyTable
}

/* 計算單字詞頻，放入原本basic node table，但遇到新的單字不增加向量維度 */
func addVecWord(newWord *node, stringBuf string, keyTable []*key) (*node, []*key) {
	memoryLimit()
	i++
	//fmt.Printf("Add word : %d\r", i)
	if newWord == nil {
		// newWord = &node{stringBuf, 1, nil}
		// keyTable = append(keyTable, &key{&newWord.key, &newWord.count})
		return newWord, keyTable
	}

	for w := newWord; w != nil; w = w.next {
		if w.key == stringBuf {
			w.count++
			return newWord, keyTable
		}
		if w.next == nil {
			// collision++
			// newNode := &node{stringBuf, 1, nil}
			// keyTable = append(keyTable, &key{&newNode.key, &newNode.count})
			// w.next = newNode
			return newWord, keyTable
		}
	}
	return newWord, keyTable
}

func printNodeTable(nodeTable []*node) {
	for n := 0; n < len(nodeTable); n++ {
		for p := nodeTable[n]; p != nil; p = p.next {
			fmt.Println(p)
		}
	}
}

func outputVector(nodeTable []*node) {
	fmt.Print("Output vector...\r")
	vecFile, err := os.OpenFile("doc2vec.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Open file fail !")
	}
	defer vecFile.Close()
	for n := 1; n < len(nodeTable); n++ {
		for p := nodeTable[n]; p != nil; p = p.next {
			vecFile.WriteString(fmt.Sprintf("%f ", p.tfidf))
		}
	}
	vecFile.WriteString(fmt.Sprintf("\n"))
}

/* 原始outputTable，會覆蓋原本檔案內容*/
// func outputNodeTable(nodeTable []*node) {
// 	fmt.Println("Output node table...")
// 	nodeTableFile, err := os.Create("NodeTable.txt")
// 	if err != nil {
// 		fmt.Println("Open file fail !")
// 	}
// 	for n := 0; n < len(nodeTable); n++ {
// 		for p := nodeTable[n]; p != nil; p = p.next {
// 			nodeTableFile.WriteString(fmt.Sprintf("%d %s", p.count, p.key))
// 		}
// 	}
// }

/* 改版outputTable，若檔案不存在則建立，檔案已存在則續寫，不會覆蓋原本內容*/
func outputNodeTable(nodeTable []*node) {
	fmt.Print("Output node table...\r")
	//nodeTableFile, err := os.Create("NodeTable.txt")
	nodeTableFile, err := os.OpenFile("NodeTable.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Open file fail !")
	}
	defer nodeTableFile.Close()
	//fmt.Println("dimension:", wordDimension)
	for n := 1; n < len(nodeTable); n++ {
		for p := nodeTable[n]; p != nil; p = p.next {
			nodeTableFile.WriteString(fmt.Sprintf("%d%s%f ", p.count, p.key, p.tfidf))
		}
	}
	nodeTableFile.WriteString(fmt.Sprintf("\n"))
}

func clearOutputFile() {
	fmt.Println("Clear NodeTable.txt & doc2vec.txt ...")
	nodeTableFile, err := os.Create("NodeTable.txt")
	vecFile, err2 := os.Create("doc2vec.txt")
	//nodeTableFile, err := os.OpenFile("NodeTable.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil || err2 != nil {
		fmt.Println("Open file fail !")
	}
	defer nodeTableFile.Close()
	defer vecFile.Close()
}

func outputSortTable(keyTable []*key) {
	fmt.Println("Output sort table...")
	SortTableFile, err := os.Create("SortTable.txt")
	if err != nil {
		fmt.Println("Open file fail !")
	}
	for k := len(keyTable) - 1; k >= 0; k-- {
		SortTableFile.WriteString(fmt.Sprintf("%d %s", *keyTable[k].count, *keyTable[k].keypos))
	}
}

func hash(a *string) uint64 {
	s := []byte(*a)
	//fmt.Println("hash number:", *a, s)
	var hashval uint64
	for i := 0; i < len(s); i++ {
		hashval = uint64(s[i]) + 131*hashval
	}
	return hashval % *args.h
}

func scanOneLine(scanner *bufio.Reader) (string, error) {
	stringBuf, readStringErr := scanner.ReadString('\n')
	//fmt.Println("scanOneLine => stringBuf:", stringBuf)
	return stringBuf, readStringErr
}

func scanOneWord(scanner *bufio.Reader) (string, error) {
	stringBuf, readStringErr := scanner.ReadString(' ')
	//stringBuf = strings.Replace(stringBuf, " ", "", -1)
	//fmt.Println("scanOneWord => stringBuf:", stringBuf)
	//s := strings.Split(stringBuf, "")
	//fmt.Println(stringBuf)
	return stringBuf, readStringErr
}

func heapsort(array []*key) {
	fmt.Println("Sorting...")
	for i := (len(array)/2 - 1); i >= 0; i-- {
		maxheap(array, i, len(array)-1)
	}
	// heapify the array into a max-heap
	for i := len(array) - 1; i > 0; i-- {
		array[0], array[i] = array[i], array[0]
		maxheap(array, 0, i-1)
	}
}

func maxheap(array []*key, start int, end int) {
	memoryLimit()
	left := start*2 + 1
	right := left + 1
	if left > end {
		return
	}
	var tmp = left
	if right <= end && *array[right].count > *array[left].count {
		tmp = right
	}
	if *array[tmp].count > *array[start].count {
		//fmt.Println(start, end, array)
		array[start].keypos, array[tmp].keypos = array[tmp].keypos, array[start].keypos
		array[start].count, array[tmp].count = array[tmp].count, array[start].count
		maxheap(array, tmp, end)
	}
}

//PrintMemUsage print memory usage
func PrintMemUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	// For info on each, see: https://golang.org/pkg/runtime/#MemStats
	fmt.Printf("Alloc = %v MiB", bToMb(m.Alloc))
	fmt.Printf("\tTotalAlloc = %v MiB", bToMb(m.TotalAlloc))
	fmt.Printf("\tSys = %v MiB", bToMb(m.Sys))
	fmt.Printf("\tNumGC = %v\n", m.NumGC)
}

func memoryLimit() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	if bToMb(m.Sys) >= *args.m {
		fmt.Println("Out of the memory limit")
		os.Exit(1)
	}
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}
