import time
from fastembed import TextEmbedding

# load documents
with open('./data/sentences.txt', 'r', encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines() if line.strip()]

def run_benchmark(model_name: str, batch_size: int = 16):
    print(f"--- Python: Loading Model {model_name} ---")
    start_load = time.time()
    embedding_model = TextEmbedding(model_name=model_name)
    print(f"Model Load Time: {time.time() - start_load:.4f}s")

    print(f"--- Starting Benchmark (Docs: {len(documents)}, Batch: {batch_size}) ---")
    start_time = time.time()
    
    # fastembed process
    embeddings = list(embedding_model.embed(documents, batch_size=batch_size))

    end_time = time.time()
    total_time = end_time - start_time
    rps = len(documents) / total_time
    
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {rps:.2f} docs/sec")
    print(f"Embedding Shape: {len(embeddings)} x {len(embeddings[0])}")
    print("-" * 30)

if __name__ == "__main__":
    # model: BAAI/bge-small-en-v1.5
    model_name = "BAAI/bge-small-en-v1.5"
    run_benchmark(model_name)
