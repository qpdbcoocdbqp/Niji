import time
import os
import click
from fastembed import TextEmbedding



# load documents
# with open('./data/sentences.txt', 'r', encoding="utf-8") as f:
#     documents = [line.strip() for line in f.readlines() if line.strip()]

@click.command()
@click.option('--input-path', '-i', default='./data/sentences.txt', help='文字檔案的路徑 (每行一句)')
@click.option('--batch-size', '-b', default=256, type=int, help='批次大小 (加大可提升處理速度)')
@click.option('--model-name', '-m', default="BAAI/bge-small-en-v1.5", help='使用的 Embedding 模型名稱')
@click.option('--parallel', '-p', default=None, type=int, help='並行處理的 Worker 數量 (預設為 None，設為 CPU 核心數可加速)')
@click.option('--threads', '-t', default=None, type=int, help='ONNX Runtime 的執行緒數量')
def main(input_path, batch_size, model_name, parallel, threads):
    # input_path check
    if not os.path.exists(input_path):
        click.echo(f"Error: File not found at {input_path}", err=True)
        return
    
    # Load documents
    click.echo(f"--- Loading data from: {input_path} ---")
    with open(input_path, 'r', encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines() if line.strip()]

    if not documents:
        click.echo("Error: Document file is empty.", err=True)
        return
    

    click.echo(f"--- Python: Loading Model {model_name} (Threads: {threads}) ---")
    start_load = time.time()
    embedding_model = TextEmbedding(model_name=model_name, threads=threads)
    click.echo(f"Model Load Time: {time.time() - start_load:.4f}s")

    click.echo(f"--- Starting Benchmark (Docs: {len(documents)}, Batch: {batch_size}, Parallel: {parallel}) ---")
    # fastembed process
    start_time = time.time()

    embeddings_generator = embedding_model.embed(
        documents, 
        batch_size=batch_size, 
        parallel=parallel
    )
    embeddings = list(embeddings_generator)
    end_time = time.time()
    total_time = end_time - start_time
    if total_time > 0:
        rps = len(documents) / total_time
    else:
        rps = 0
    
    click.echo(f"Total Time: {total_time:.4f}s")
    click.echo(f"Throughput: {rps:.2f} docs/sec")
    
    if embeddings:
        click.echo(f"Embedding Shape: {len(embeddings)} x {len(embeddings[0])}")
    
    click.echo("-" * 30)

if __name__ == "__main__":
    main()
