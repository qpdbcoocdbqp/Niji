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
  
    Issue: `1B` model load to CUDA still OOM.
    Use `270m` model can load to CUDA.


    ```sh
    cd candle-gemma

    # build
    cargo build --example quantized-gemma --release --features cuda

    # generate
    ./target/release/examples/quantized-gemma \
    --model $MODEL_HOME/huggingface/hub/models--ggml-org--gemma-3-270m-it-GGUF/snapshots/e7647be17ae1108f2f605ed061ca0608b171afff/gemma-3-270m-it-Q8_0.gguf \
    --tokenizer $MODEL_HOME/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3/tokenizer.json \
    --top-p 0.9 \
    --repeat-penalty 2.0 \
    --repeat-last-n 512 \
    --temperature 0.7 \
    --context-len 512 \
    --sample-len 128 \
    --prompt 'Here is a proof that square root of 2 is not rational: '
    ```
