package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/anush008/fastembed-go"
)

// Global variables for command line arguments
var (
	documentPath string
	batchSize    int
	threads      int
	cores        int
)

func init() {
	// 1. Define flags in English
	flag.StringVar(&documentPath, "path", "../data/sentences.txt", "Path to the input file containing sentences")
	flag.IntVar(&batchSize, "batch_size", 64, "Batch size for encoding (Recommended: 64, 128, 256)")

	// Default threads set to 0 means letting ONNX Runtime decide (usually uses all available cores)
	flag.IntVar(&threads, "threads", 0, "Number of threads for ONNX Runtime (intra-op parallelism)")

	// Default Go cores
	flag.IntVar(&cores, "cores", runtime.NumCPU(), "Number of CPU cores used by Go runtime (GOMAXPROCS)")

	flag.Parse()
}

// setEnvironmentVariables configures the underlying linear algebra libraries
func setEnvironmentVariables(numThreads int) {
	if numThreads > 0 {
		sThreads := fmt.Sprintf("%d", numThreads)
		// OpenMP threads (Common for ONNX)
		os.Setenv("OMP_NUM_THREADS", sThreads)
		// Intel MKL threads (Crucial for Intel CPUs)
		os.Setenv("MKL_NUM_THREADS", sThreads)
		// General ONNX threads
		os.Setenv("ONNX_NUM_THREADS", sThreads)
		fmt.Printf("Configured Environment: OMP/MKL_NUM_THREADS=%s\n", sThreads)
	}
}

func runBenchmark(modelName string, documents []string, batchSize int) {
	fmt.Printf("--- Loading Model: %s ---\n", modelName)
	startLoad := time.Now()

	// Initialize options
	// Note: 'Threads' in InitOptions might not be fully supported by all wrapper versions,
	// so we rely on environment variables set in main() for threading control.
	options := fastembed.InitOptions{
		Model:     fastembed.BGESmallENV15,
		CacheDir:  "model_cache",
		MaxLength: 512,
	}

	model, err := fastembed.NewFlagEmbedding(&options)
	if err != nil {
		panic(fmt.Errorf("failed to load model: %w", err))
	}
	defer model.Destroy()

	loadTime := time.Since(startLoad).Seconds()
	fmt.Printf("Model Loaded in: %.4fs\n", loadTime)

	fmt.Printf("--- Starting Benchmark ---\n")
	fmt.Printf("Docs: %d | Batch Size: %d\n", len(documents), batchSize)

	start := time.Now()

	// Execute Embedding
	// The library handles batching internally based on the batchSize parameter
	_, err = model.Embed(documents, batchSize)
	if err != nil {
		panic(fmt.Errorf("embedding failed: %w", err))
	}

	duration := time.Since(start).Seconds()

	// Avoid division by zero
	if duration < 1e-9 {
		duration = 1e-9
	}

	rps := float64(len(documents)) / duration

	fmt.Println("------------------------------")
	fmt.Printf("Total Processing Time: %.4fs\n", duration)
	fmt.Printf("Throughput:            %.2f docs/sec\n", rps)
	fmt.Println("------------------------------")
}

func main() {
	// 1. Optimize Runtime & Environment BEFORE any heavy lifting
	if cores > 0 {
		runtime.GOMAXPROCS(cores)
	}
	setEnvironmentVariables(threads)

	// 2. Read File
	file, err := os.Open(documentPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: Unable to open file '%s': %v\n", documentPath, err)
		os.Exit(1)
	}
	defer file.Close()

	var documents []string
	scanner := bufio.NewScanner(file)

	// Create a buffer for scanner to handle long lines if necessary
	// const maxCapacity = 1024 * 1024 // 1MB
	// buf := make([]byte, maxCapacity)
	// scanner.Buffer(buf, maxCapacity)

	for scanner.Scan() {
		text := scanner.Text()
		if text != "" {
			documents = append(documents, text)
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
	}

	if len(documents) == 0 {
		fmt.Printf("Warning: No documents found in %s\n", documentPath)
		return
	}

	// 3. Run Benchmark
	runBenchmark("BAAI/bge-small-en-v1.5", documents, batchSize)
}
