// https://github.com/huggingface/candle/blob/689d255b/candle-transformers/src/models/quantized_gemma3.rs#L261-L269
//! Gemma 3 model implementation with quantization support.
//! 

use candle::quantized::gguf_file;
use candle::quantized::QTensor;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::Module;
// use candle_transformers::utils;
use candle_transformers::quantized_nn::RmsNorm;
use candle::D;

pub const MAX_SEQ_LEN: usize = 4096; // Gemma 3 supports 128K context window
pub const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;
// pub const DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR: f32 = 1.;


#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul,
    feed_forward_up: QMatMul,
    feed_forward_down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_start = std::time::Instant::now();
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        log::trace!("  MLP gate/up: {:.2}ms", gate_start.elapsed().as_secs_f64() * 1000.0);
        
        let act_start = std::time::Instant::now();
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        log::trace!("  MLP activation: {:.2}ms", act_start.elapsed().as_secs_f64() * 1000.0);
        
        let down_start = std::time::Instant::now();
        let result = self.feed_forward_down.forward(&gated)?;
        log::trace!("  MLP down: {:.2}ms", down_start.elapsed().as_secs_f64() * 1000.0);
        
        Ok(result)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    // Attention components
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,

    // Specialized normalization for Q and K
    attention_q_norm: RmsNorm,
    attention_k_norm: RmsNorm,

    // Layer normalization
    attention_norm: RmsNorm,      // Applied before attention
    post_attention_norm: RmsNorm, // Applied after attention
    ffn_norm: RmsNorm,            // Applied before feedforward
    post_ffn_norm: RmsNorm,       // Applied after feedforward

    // Feed-forward network
    mlp: Mlp,

    // Attention parameters
    n_head: usize,    // Number of query heads
    n_kv_head: usize, // Number of key-value heads
    head_dim: usize,  // Dimension of each head
    q_dim: usize,     // Total dimension for queries

    sliding_window_size: Option<usize>,

    rotary_embedding: RotaryEmbedding,
    neg_inf: Tensor,

    // Cache
    kv_cache: Option<(Tensor, Tensor)>,

    // Tracing
    span_attn: tracing::Span,
    span_mlp: tracing::Span,
}

impl LayerWeights {
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sliding_window_size < i {
                            0u32
                        } else {
                            1u32
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask = if index_pos > 0 {
            let mask0 = Tensor::zeros((seq_len, index_pos), DType::F32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
            .to_dtype(dtype)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        let qkv_start = std::time::Instant::now();
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;
        log::trace!("  QKV projection: {:.2}ms", qkv_start.elapsed().as_secs_f64() * 1000.0);

        let reshape_start = std::time::Instant::now();
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        log::trace!("  Reshape/transpose: {:.2}ms", reshape_start.elapsed().as_secs_f64() * 1000.0);

        let norm_start = std::time::Instant::now();
        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;
        log::trace!("  Q/K norm: {:.2}ms", norm_start.elapsed().as_secs_f64() * 1000.0);

        let rope_start = std::time::Instant::now();
        let (q, k) = self
            .rotary_embedding
            .apply_rotary_emb_qkv(&q, &k, index_pos)?;
        log::trace!("  RoPE: {:.2}ms", rope_start.elapsed().as_secs_f64() * 1000.0);

        let cache_start = std::time::Instant::now();
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?; // concat on seq dim
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone())); // update cache
        log::trace!("  KV cache: {:.2}ms", cache_start.elapsed().as_secs_f64() * 1000.0);

        let repeat_start = std::time::Instant::now();
        // Repeat KV for GQA
        let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;
        log::trace!("  Repeat KV: {:.2}ms", repeat_start.elapsed().as_secs_f64() * 1000.0);

        let attn_start = std::time::Instant::now();
        // Scaled Dot-Product Attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        log::trace!("  Attention computation: {:.2}ms", attn_start.elapsed().as_secs_f64() * 1000.0);

        let output_start = std::time::Instant::now();
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?;

        let result = self.attention_wo.forward(&attn_output)?;
        log::trace!("  Output projection: {:.2}ms", output_start.elapsed().as_secs_f64() * 1000.0);
        
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    weight: Tensor,
}

impl QuantizedEmbedding {
    pub fn new(weight: QTensor, dev: &Device) -> Result<Self> {
        let weight = weight.dequantize(dev)?;
        Ok(Self { weight })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        let res = self.weight.index_select(&x.reshape(b_sz * seq_len)?, 0)?;
        res.reshape((b_sz, seq_len, ()))
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: QuantizedEmbedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        _kv_cache_dtype: DType,
        _context_len: usize,
    ) -> Result<Self> {
        let total_start = std::time::Instant::now();
        log::info!("Starting model loading from GGUF...");

        let metadata_start = std::time::Instant::now();
        let prefix = ["gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
            })
            .copied()
            .unwrap_or("gemma3");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize;
        let _value_length = md_get("attention.value_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("attention.sliding_window")?.to_u32()? as usize;

        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        let rope_freq_base = md_get("gemma3.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let rope_freq_base_sliding = md_get("gemma3.rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);
        
        let q_dim = head_count * key_length;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        log::info!(
            "Metadata parsed in {:.2}ms: {} layers, {} heads, {} kv_heads, dim={}, head_dim={}",
            metadata_start.elapsed().as_secs_f64() * 1000.0,
            block_count, head_count, head_count_kv, embedding_length, key_length
        );

        let emb_start = std::time::Instant::now();
        let tok_emb_qtensor = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = QuantizedEmbedding::new(tok_emb_qtensor, device)?;
        log::info!("Token embeddings loaded in {:.2}ms", emb_start.elapsed().as_secs_f64() * 1000.0);

        let norm_start = std::time::Instant::now();
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?,
        };
        log::info!("Output norm and weights loaded in {:.2}ms", norm_start.elapsed().as_secs_f64() * 1000.0);

        log::info!("Loading {} transformer layers...", block_count);
        let layers_start = std::time::Instant::now();
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let layer_start = std::time::Instant::now();
            let prefix = format!("blk.{layer_idx}");

            let attn_weights_start = std::time::Instant::now();
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            log::trace!("  Layer {} attention weights: {:.2}ms", layer_idx, attn_weights_start.elapsed().as_secs_f64() * 1000.0);

            let norms_start = std::time::Instant::now();
            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(
                    reader,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )?,
                rms_norm_eps,
            )?;
            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            log::trace!("  Layer {} norms: {:.2}ms", layer_idx, norms_start.elapsed().as_secs_f64() * 1000.0);

            let mlp_start = std::time::Instant::now();
            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?)?,
                feed_forward_up: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?)?,
                feed_forward_down: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?)?,
            };
            log::trace!("  Layer {} MLP weights: {:.2}ms", layer_idx, mlp_start.elapsed().as_secs_f64() * 1000.0);

            let rope_start = std::time::Instant::now();
            // Sliding window pattern hardcoded to 6 because it's not explicitly defined
            let is_sliding = (layer_idx + 1) % sliding_window_type > 0;
            let sliding_window_size = is_sliding.then_some(sliding_window_size);
            let layer_rope_frequency = if is_sliding {
                rope_freq_base_sliding
            } else {
                rope_freq_base
            };
            
            let rotary_embedding = RotaryEmbedding::new(key_length, layer_rope_frequency, device)?;
            log::trace!("  Layer {} RoPE: {:.2}ms", layer_idx, rope_start.elapsed().as_secs_f64() * 1000.0);

            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "mlp");

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: key_length,
                q_dim,
                sliding_window_size,
                rotary_embedding,
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_mlp,
            });
            
            if (layer_idx + 1) % 5 == 0 || layer_idx == block_count - 1 {
                log::info!(
                    "Loaded {}/{} layers (last layer took {:.2}ms)",
                    layer_idx + 1, block_count,
                    layer_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        log::info!("All {} layers loaded in {:.2}s", block_count, layers_start.elapsed().as_secs_f64());

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        log::info!("Model loading completed in {:.2}s", total_start.elapsed().as_secs_f64());

        Ok(Self {
            tok_embeddings,
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            span,
            span_output,
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let start_time = std::time::Instant::now();
        let (b_sz, seq_len) = x.dims2()?;
        let _enter = self.span.enter();

        log::debug!("Forward pass started: batch_size={}, seq_len={}, index_pos={}", b_sz, seq_len, index_pos);

        let emb_start = std::time::Instant::now();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;
        log::debug!("Token embedding took: {:.2}ms", emb_start.elapsed().as_secs_f64() * 1000.0);

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_start = std::time::Instant::now();
            
            let mask_start = std::time::Instant::now();
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, x.dtype(), x.device())?)
            };
            let mask_time = mask_start.elapsed().as_secs_f64() * 1000.0;

            // Attention block
            let attn_start = std::time::Instant::now();
            let residual = &layer_in;
            
            let norm_start = std::time::Instant::now();
            let x = layer.attention_norm.forward(&layer_in)?;
            let norm1_time = norm_start.elapsed().as_secs_f64() * 1000.0;
            
            let attn_forward_start = std::time::Instant::now();
            let x = layer.forward_attn(&x, attention_mask.as_ref(), index_pos)?;
            let attn_forward_time = attn_forward_start.elapsed().as_secs_f64() * 1000.0;
            
            let post_norm_start = std::time::Instant::now();
            let x = layer.post_attention_norm.forward(&x)?;
            let post_norm_time = post_norm_start.elapsed().as_secs_f64() * 1000.0;
            
            let x = (x + residual)?;
            let attn_time = attn_start.elapsed().as_secs_f64() * 1000.0;

            // Feed-forward block
            let mlp_start = std::time::Instant::now();
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            
            let ffn_norm_start = std::time::Instant::now();
            let x = layer.ffn_norm.forward(&x)?;
            let ffn_norm_time = ffn_norm_start.elapsed().as_secs_f64() * 1000.0;
            
            let mlp_forward_start = std::time::Instant::now();
            let x = layer.mlp.forward(&x)?;
            let mlp_forward_time = mlp_forward_start.elapsed().as_secs_f64() * 1000.0;
            
            let post_ffn_start = std::time::Instant::now();
            let x = layer.post_ffn_norm.forward(&x)?;
            let post_ffn_time = post_ffn_start.elapsed().as_secs_f64() * 1000.0;
            
            let x = (x + residual)?;
            drop(_enter);
            let mlp_time = mlp_start.elapsed().as_secs_f64() * 1000.0;

            layer_in = x;
            
            let layer_time = layer_start.elapsed().as_secs_f64() * 1000.0;
            log::debug!(
                "Layer {}: {:.2}ms (mask: {:.2}ms, attn: {:.2}ms [norm: {:.2}ms, forward: {:.2}ms, post_norm: {:.2}ms], mlp: {:.2}ms [norm: {:.2}ms, forward: {:.2}ms, post_norm: {:.2}ms])",
                layer_idx, layer_time, mask_time, attn_time, norm1_time, attn_forward_time, post_norm_time, 
                mlp_time, ffn_norm_time, mlp_forward_time, post_ffn_time
            );
        }

        let _enter = self.span_output.enter();

        let output_start = std::time::Instant::now();
        let x = layer_in.i((.., seq_len - 1, ..))?;
        let x = self.norm.forward(&x)?;
        let output = self.output.forward(&x)?;
        log::debug!("Output layer took: {:.2}ms", output_start.elapsed().as_secs_f64() * 1000.0);

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        log::info!("Total forward pass took: {:.2}ms", total_time);

        Ok(output)
    }
}
