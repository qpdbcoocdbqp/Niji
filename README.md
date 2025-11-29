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

- Rust: [fastembed-rs](https://github.com/Anush008/fastembed-rs)
- Python: [fastembed](https://github.com/qdrant/fastembed)
- Go: [fastembed-go](https://github.com/Anush008/fastembed-go)

## Test enviroment

- **python**

```sh
cd python
docker build -t niji:python -f Dockerfile .
```

- **rust**

```sh
cd rust
docker build -t niji:rust -f Dockerfile .

docker run -d \
-v .//data://app//data \
-v .//rust//.fastembed_cache://app//.fastembed_cache \
--name nj-rust \
niji:rust sleep infinity
docker exec -it nj-rust ./rust_benchmark
```
