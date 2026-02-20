## Candle

* **IMPORTANT**: Here use Windows and run in `wsl`.

* Prerequirememts:

    ```sh
    # check cuda is installed
    export PATH=$WSL_HOST/cuda-12.9/bin:$PATH
    export LD_LIBRARY_PATH=$WSL_HOST/cuda-12.9/targets/x86_64-linux/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH

    nvcc --version
    nvidia-smi
    ```

* Examples
  * load from `safetensors`

    ```sh
    cd candle-gemma

    # build
    cargo build --example gemma --release --features cuda

    # generate
    ./target/release/examples/gemma \
    --tokenizer-file $MODEL_HOME/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3/tokenizer.json \
    --config-file $MODEL_HOME/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3/config.json \
    --weight-files $MODEL_HOME/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3/model.safetensors \
    --top-p 0.9 \
    --repeat-penalty 2.0 \
    --temperature 0.7 \
    --sample-len 128 \
    --prompt 'Here is a proof that square root of 2 is not rational: '
    ```

  * load from `gguf`

    The script can test `1B`, `270M` model on GPU.
    
    ```sh
    cd candle-gemma

    # build
    cargo build --example gemma-gguf --release --features cuda

    # generate
    RUST_LOG=info ./target/release/examples/gemma-gguf \
    --model $MODEL_HOME/huggingface/hub/models--google--gemma-3-1b-it-qat-q4_0-gguf/snapshots/d1be121d36172a4b0b964657e2ee859d61138593/gemma-3-1b-it-q4_0.gguf \
    --tokenizer $MODEL_HOME/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752/tokenizer.json \
    --top-p 0.9 \
    --repeat-penalty 2.0 \
    --repeat-last-n 512 \
    --temperature 0.7 \
    --context-len 512 \
    --sample-len 128 \
    --kv-cache-type f16 \
    --prompt 'Here is a proof that square root of 2 is not rational: '
    ```
