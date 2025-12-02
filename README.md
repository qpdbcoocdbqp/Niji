# Niji
Jam fastembed with python, rust and golang. Playing with [虹](https://www.youtube.com/watch?v=SXyqhjhaQQA)

- **About 虹 / にじ (Niji)**

> 虹 · 福山雅治 · Gonen Mono
> 
> from 18th SINGLE『虹/ひまわり/それがすべてさ』
> 
> 2003/08/27 Release

  - **Recommend**:
    - [福山雅治 - 虹 (【男性限定LIVE】福山☆冬の大感謝祭 其の十七 野郎夜!!4)](https://www.youtube.com/watch?v=Ws0jIc3Cmxk)

## Reference

| Language | Repository |
| --- | --- |
| Rust | [fastembed-rs](https://github.com/Anush008/fastembed-rs) |
| Python | [fastembed](https://github.com/qdrant/fastembed) |
| Go | [fastembed-go](https://github.com/Anush008/fastembed-go) |

## Test enviroment

- Testing data senctance is from [Donald Trump - Wikipedia](https://en.wikipedia.org/wiki/Donald_Trump)

### **PYTHON**

- model source: [Qdrant/bge-small-en-v1.5-onnx-Q](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q/tree/main)
    ```sh
    cd python
    docker build -t niji:python -f Dockerfile .

    docker run -d \
    --cpus 1 \
    -v ./data:/app/data \
    -v ~/.cache/huggingface/hub:/tmp/fastembed_cache \
    --name nj-py \
    niji:python sleep infinity

    docker exec -it nj-py python -m main -b 16 -p 1 -t 1 -i ./data/sentences.txt -m BAAI/bge-small-en-v1.5
    docker exec -it nj-py python -m main -b 256 -p 1 -t 1 -i ./data/sentences.txt -m BAAI/bge-small-en-v1.5
    docker exec -it nj-py python -m main -b 1024 -p 1 -t 1 -i ./data/sentences.txt -m BAAI/bge-small-en-v1.5

    docker rm -f nj-py
    ```

### **RUST**

- model source: [Xenova/bge-small-en-v1.5](Xenova/bge-small-en-v1.5)

    ```sh
    cd rust
    docker build -t niji:rust -f Dockerfile .

    docker run -d \
    --cpus 1 \
    -v ./data:/app/data \
    -v ~/.cache/huggingface/hub:/app/.fastembed_cache \
    --name nj-rs \
    niji:rust sleep infinity

    docker exec -it nj-rs ./rust_benchmark ./data/sentences.txt 16 1 1
    docker exec -it nj-rs ./rust_benchmark ./data/sentences.txt 256 1 1
    docker exec -it nj-rs ./rust_benchmark ./data/sentences.txt 1024 1 1

    docker rm -f nj-rs
    ```

### **GO**

- model souece: `fast-bge-small-en-v1.5` (unknown source)

    ```sh
    cd go
    docker build -t niji:go -f Dockerfile .

    docker run -d \
    --cpus 4 \
    -v ./data:/app/data \
    -v ~/.cache/huggingface/hub:/app/model_cache \
    --name nj-go \
    niji:go sleep infinity

    docker exec -it nj-go ./benchmark -path=./data/sentences.txt -batch_size=16 -cores=4 -threads=4
    docker exec -it nj-go ./benchmark -path=./data/sentences.txt -batch_size=256 -cores=4 -threads=4
    docker exec -it nj-go ./benchmark -path=./data/sentences.txt -batch_size=1024 -cores=4 -threads=4

    docker rm -f nj-go
    ```

## Test Result

- CPU: Intel(R) Core(TM) i7-14650HX CPU
- RAM: unlimit in this test
- sentences: 632

| language | batch_size| cores | threads | time (second) | throughput (sentences/sec) |
| --- | --- | --- | --- | --- | --- |
| Python | 16 | 1 | 1 | 14.8374 | 42.59 |
| Python | 256 | 1 | 1 | 24.0746 | 26.25 |
| Python | 1024 | 1 | 1 | 23.2741 | 27.15 |
| Rust | 16 | 1 | 1 | 11.7152 | 53.95 |
| Rust | 256 | 1 | 1 | 18.7973 | 33.62 |
| Rust | 1024 | 1 | 1 | 21.9385 | 28.81 |
| Go | 16 | 4 | 4 | 120.0003 | 5.27 |
| Go | 256 | 4 | 4 | 31.5784 | 20.01 |
| Go | 1024 | 4 | 4 | 32.8536 | 19.24 |

