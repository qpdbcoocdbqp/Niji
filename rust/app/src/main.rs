use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::fs;
use std::time::Instant;


fn run_benchmark(documents: &Vec<String>) {        
    println!("--- Rust: Loading Model BAAI/bge-small-en-v1.5 ---");
    let mut model: TextEmbedding = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
    ).expect("Failed to initialize TextEmbedding model");

    println!("--- Starting Benchmark (Docs: {}) ---", documents.len());
    let start_time: Instant = Instant::now();
    // Generate embeddings with the default batch size, 256
    let embeddings: Vec<Vec<f32>> = model.embed(documents.clone(), Some(16))
        .expect("Failed to embeddings");

    let duration: std::time::Duration = start_time.elapsed();
    let rps: f64 = documents.len() as f64 / duration.as_secs_f64();
    println!("Total Time: {:.4}s", duration.as_secs_f64());
    println!("Throughput: {:.2} docs/sec", rps);
    println!("Embeddings length: {}", embeddings.len());
    println!("Embedding dimension: {}", embeddings[0].len());
    println!("------------------------------");
}

fn main() {
    let content = fs::read_to_string("./data/sentences.txt").expect("Failed to read file");
    let documents: Vec<String> = content.lines().map(|s| s.to_string()).collect();

    run_benchmark(&documents);
}


