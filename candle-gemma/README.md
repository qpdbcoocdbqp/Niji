## Candle

* **IMPORTANT**: Here use Windows and run in `wsl`.

```sh
# check cuda is installed
export PATH=$WSL_HOST/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=$WSL_HOST/cuda-12.9/targets/x86_64-linux/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH

nvcc --version
nvidia-smi

# build candle for gemma case
cargo new candle-gemma && cd candle-gemma
cargo add --git https://github.com/huggingface/candle.git candle-core --features cuda
cargo add --git https://github.com/huggingface/candle.git candle-transformers --features cuda
cargo add --git https://github.com/huggingface/candle.git candle-examples --features cuda
cargo build --example gemma --release --features cuda

# run candle for gemma case
./target/release/examples/gemma \
--tokenizer-file $MODEL_HOST/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3/tokenizer.json \
--config-file $MODEL_HOST/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3/config.json \
--weight-files $MODEL_HOST/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3/model.safetensors \
--repeat-penalty 2.0 \
--sample-len 128 \
--temperature 0.7 \
--top-p 0.9 \
--prompt 'Here is a proof that square root of 2 is not rational: '
```
