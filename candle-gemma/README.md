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
    --tokenizer-file $MODEL_HOST/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3/tokenizer.json \
    --config-file $MODEL_HOST/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3/config.json \
    --weight-files $MODEL_HOST/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3/model.safetensors \
    --repeat-penalty 2.0 \
    --sample-len 128 \
    --temperature 0.7 \
    --top-p 0.9 \
    --prompt 'Here is a proof that square root of 2 is not rational: '
    ```

  * load from `gguf`
  
    Issue: Load to CUDA still OOM.

    ```sh
    cd candle-gemma

    # build
    cargo build --example quantized-gemma --release --features cuda

    # generate
    ./target/release/examples/quantized-gemma \
    --model $MODEL_HOME/huggingface/hub/models--google--gemma-3-1b-it-qat-q4_0-gguf/snapshots/d1be121d36172a4b0b964657e2ee859d61138593/gemma-3-1b-it-q4_0.gguf \
    --tokenizer $MODEL_HOME/huggingface/hub/models--google--gemma-3-1b-it-qat-q4_0-unquantized/snapshots/a6692c1945954f4aa39a17b8dfba4a7e62db3d4f/tokenizer.json \
    --prompt 'Here is a proof that square root of 2 is not rational: ' \
    --context-len 512 \
    --sample-len 128 \
    --temperature 0.7 \
    --top-p 0.9 \
    --repeat-penalty 2.0 \
    --kv-cache-type f16 \
    --cpu
    ```
