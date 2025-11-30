use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn run_benchmark(documents: &[String],  batch_size: usize) {
    println!("--- Rust: Loading Model BAAI/bge-small-en-v1.5 ---");
    
    // init mode;
    let mut model: TextEmbedding = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
    )
    .expect("Failed to initialize TextEmbedding model");

    println!("--- Starting Benchmark (Docs: {}) ---", documents.len());
    println!("Batch Size: {}", batch_size);
    println!(
        "Config: Rayon Threads (Data) = {}, OMP Threads (Math) = {}",
        env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "Default".to_string()),
        env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "Default".to_string())
    );
    
    let start_time: Instant = Instant::now();
    let document_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
    let embeddings = model
        .embed(document_refs, Some(batch_size))
        .expect("Failed to generate embeddings");
    let duration: std::time::Duration = start_time.elapsed();
    let rps: f64 = documents.len() as f64 / duration.as_secs_f64();
    
    println!("Total Time: {:.4}s", duration.as_secs_f64());
    println!("Throughput: {:.2} docs/sec", rps);
    
    if let Some(first_embed) = embeddings.first() {
        println!("Embeddings length: {}", embeddings.len());
        println!("Embedding dimension: {}", first_embed.len());
    }
    println!("------------------------------");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    // default_path use `./data/sentences.txt`
    let default_path = "./data/sentences.txt";
    let file_path_str = args.get(1).map(|s| s.as_str()).unwrap_or(default_path);
    let path = Path::new(file_path_str);
    println!("Reading file from: {}", file_path_str);

    let content = fs::read_to_string(path).unwrap_or_else(|_| {
        panic!("Failed to read file at: {}", file_path_str)
    });
    let documents: Vec<String> = content.lines().map(|s| s.to_string()).collect();

    if documents.is_empty() {
        println!("Warning: The document list is empty.");
        return;
    }

    // default batch_size is 256
    let batch_size = args.get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256);

    // parse Threads (control Rayon/data processing parallelism)
    if let Some(threads) = args.get(3) {
        env::set_var("RAYON_NUM_THREADS", threads);
    }

    // parse Cores (control ONNX/matrix operation core number)
    if let Some(cores) = args.get(4) {
        env::set_var("OMP_NUM_THREADS", cores);
        env::set_var("MKL_NUM_THREADS", cores);
    }

    run_benchmark(&documents, batch_size);
}