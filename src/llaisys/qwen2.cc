#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"   
#include "../utils/check.hpp"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdio.h>

 struct LlaisysQwen2Model{
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights w;

    llaisysTensor_t* cache_k;
    llaisysTensor_t* cache_v;
    size_t cur_pos;//cahce到第几个token

    // ---- scratch for 1-layer attention ----
    llaisysTensor_t s_tokens;     // [maxseq] I64
    llaisysTensor_t s_pos;        // [maxseq] I64, 内容=0..maxseq-1

    llaisysTensor_t s_x0;         // [maxseq, hs] bf16  (embedding output)
    llaisysTensor_t s_x1;         // [maxseq, hs] bf16  (rmsnorm output)

    llaisysTensor_t s_qflat;      // [maxseq, hs] bf16
    llaisysTensor_t s_kflat;      // [maxseq, kv] bf16   kv = nkvh*dh
    llaisysTensor_t s_vflat;      // [maxseq, kv] bf16

    llaisysTensor_t s_attn;       // [maxseq, nh, dh] bf16 (self_attention output)
    llaisysTensor_t s_attn_flat;  // view: [maxseq, hs] bf16

    llaisysTensor_t s_o;          // [maxseq, hs] bf16 (o_proj output)
    llaisysTensor_t s_o_bias;     // [hs] bf16 zeros (linear 需要 bias)

    llaisysTensor_t s_x2;         // [maxseq, hs] bf16 (residual x0 + o)

    // last token / logits / argmax 
    llaisysTensor_t s_last;       // [1, hs] bf16  (slice view)
    llaisysTensor_t s_last_norm;  // [1, hs] bf16
    llaisysTensor_t s_logits;     // [1, voc] bf16
    llaisysTensor_t s_logits_1d;  // view [voc]
    llaisysTensor_t s_arg_i;      // [1] I64
    llaisysTensor_t s_arg_v;      // [1] bf16
    llaisysTensor_t s_lm_bias;    // [voc] bf16 zeros

    llaisysTensor_t s_x3;
    llaisysTensor_t s_x2_norm;
    llaisysTensor_t s_gate;
    llaisysTensor_t s_up;
    llaisysTensor_t s_act;
    llaisysTensor_t s_down;
    llaisysTensor_t s_gate_bias;
    llaisysTensor_t s_up_bias;
    llaisysTensor_t s_down_bias;
 };
 

static inline llaisysTensor_t* alloc_tensor_array(size_t nlayer) {
    return (llaisysTensor_t*)std::malloc(sizeof(llaisysTensor_t) * nlayer);
}

static inline void zero_tensor_array(llaisysTensor_t* p, size_t nlayer) {
    std::memset(p, 0, sizeof(llaisysTensor_t) * nlayer);
}

extern "C"{

__export LlaisysQwen2Model* llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta* meta,
    llaisysDeviceType_t device,
    int* device_ids,
    int ndevice
){
    (void)ndevice;
    int dev_id = (device_ids ? device_ids[0] : 0);

    auto* model = (LlaisysQwen2Model*)std::malloc(sizeof(LlaisysQwen2Model));
    std::memset(model, 0, sizeof(LlaisysQwen2Model));
    model->meta = *meta;

    //初始化weights
    //in_embed out_embed
    {
        size_t shape[2] = {meta->voc, meta->hs};
        model->w.in_embed = tensorCreate(shape, 2, meta->dtype, device, dev_id); 
        model->w.out_embed = tensorCreate(shape, 2, meta->dtype, device, dev_id);
    }
    //out_norm_W
    {
        size_t shape[1] = {meta->hs};
        model->w.out_norm_w = tensorCreate(shape, 1, meta->dtype, device, dev_id);
    }

    const size_t L = meta->nlayer;
    const size_t HS = meta->hs;
    const size_t DI = meta->di;
    const size_t KV = meta->nkvh * meta->dh; 

    model->w.attn_norm_w = alloc_tensor_array(L);
    model->w.attn_q_w     = alloc_tensor_array(L);
    model->w.attn_q_b     = alloc_tensor_array(L);
    model->w.attn_k_w     = alloc_tensor_array(L);
    model->w.attn_k_b     = alloc_tensor_array(L);
    model->w.attn_v_w     = alloc_tensor_array(L);
    model->w.attn_v_b     = alloc_tensor_array(L);
    model->w.attn_o_w     = alloc_tensor_array(L);

    model->w.mlp_norm_w   = alloc_tensor_array(L);
    model->w.mlp_gate_w   = alloc_tensor_array(L);
    model->w.mlp_up_w     = alloc_tensor_array(L);
    model->w.mlp_down_w   = alloc_tensor_array(L);

    
    zero_tensor_array(model->w.attn_norm_w, L);
    zero_tensor_array(model->w.attn_q_w, L);
    zero_tensor_array(model->w.attn_q_b, L);
    zero_tensor_array(model->w.attn_k_w, L);
    zero_tensor_array(model->w.attn_k_b, L);
    zero_tensor_array(model->w.attn_v_w, L);
    zero_tensor_array(model->w.attn_v_b, L);
    zero_tensor_array(model->w.attn_o_w, L);

    zero_tensor_array(model->w.mlp_norm_w, L);
    zero_tensor_array(model->w.mlp_gate_w, L);
    zero_tensor_array(model->w.mlp_up_w, L);
    zero_tensor_array(model->w.mlp_down_w, L);

    // -------- per-layer tensorCreate --------
    for (size_t i = 0; i < L; ++i) {
        // layer norms: [hs]
        {
            size_t s[1] = { HS };
            model->w.attn_norm_w[i] = tensorCreate(s, 1, meta->dtype, device, dev_id);
            model->w.mlp_norm_w[i]  = tensorCreate(s, 1, meta->dtype, device, dev_id);
        }

        // q proj: weight [hs, hs], bias [hs]
        {
            size_t sw[2] = { HS, HS };
            size_t sb[1] = { HS };
            model->w.attn_q_w[i] = tensorCreate(sw, 2, meta->dtype, device, dev_id);
            model->w.attn_q_b[i] = tensorCreate(sb, 1, meta->dtype, device, dev_id);
        }

        // k/v proj: weight [KV, hs], bias [KV]
        {
            size_t sw[2] = { KV, HS };
            size_t sb[1] = { KV };
            model->w.attn_k_w[i] = tensorCreate(sw, 2, meta->dtype, device, dev_id);
            model->w.attn_k_b[i] = tensorCreate(sb, 1, meta->dtype, device, dev_id);
            model->w.attn_v_w[i] = tensorCreate(sw, 2, meta->dtype, device, dev_id);
            model->w.attn_v_b[i] = tensorCreate(sb, 1, meta->dtype, device, dev_id);
        }

        // o proj: weight [hs, hs]（没有 bias）
        {
            size_t sw[2] = { HS, HS };
            model->w.attn_o_w[i] = tensorCreate(sw, 2, meta->dtype, device, dev_id);
        }

        // MLP: gate/up [di, hs], down [hs, di]
        {
            size_t su[2] = { DI, HS };
            size_t sd[2] = { HS, DI };
            model->w.mlp_gate_w[i] = tensorCreate(su, 2, meta->dtype, device, dev_id);
            model->w.mlp_up_w[i]   = tensorCreate(su, 2, meta->dtype, device, dev_id);
            model->w.mlp_down_w[i] = tensorCreate(sd, 2, meta->dtype, device, dev_id);
        }
    }

    // after per-layer weight tensors created ...
    {
        model->cur_pos = 0;

        const size_t L  = meta->nlayer;
        const size_t MS = meta->maxseq;
        const size_t KVH = meta->nkvh;
        const size_t DH = meta->dh;
        const size_t HS = meta->hs;
        const size_t NH = meta->nh;
        const size_t KV = meta->nkvh * meta->dh; // 256
        const size_t VOC = meta->voc;
        const size_t DI = meta->di;

        model->cache_k = (llaisysTensor_t*)std::malloc(sizeof(llaisysTensor_t) * L);
        model->cache_v = (llaisysTensor_t*)std::malloc(sizeof(llaisysTensor_t) * L);

        for (size_t i = 0; i < L; ++i) {
            size_t s[3] = { MS, KVH, DH };
            model->cache_k[i] = tensorCreate(s, 3, meta->dtype, device, dev_id);
            model->cache_v[i] = tensorCreate(s, 3, meta->dtype, device, dev_id);
        }

        // tokens: [maxseq] I64
        {
            size_t s[1] = { MS };
            model->s_tokens = tensorCreate(s, 1, LLAISYS_DTYPE_I64, device, dev_id);
        }

        // pos ids: [maxseq] I64, fill 0..MS-1 once
        {
            size_t s[1] = { MS };
            model->s_pos = tensorCreate(s, 1, LLAISYS_DTYPE_I64, device, dev_id);

            // host fill
            int64_t* host = (int64_t*)std::malloc(sizeof(int64_t) * MS);
            for (size_t i = 0; i < MS; ++i) host[i] = (int64_t)i;
            tensorLoad(model->s_pos, host);
            std::free(host);
        }

        // x0/x1/x2: [maxseq, hs]
        {
            size_t s[2] = { MS, HS };
            model->s_x0 = tensorCreate(s, 2, meta->dtype, device, dev_id);
            model->s_x1 = tensorCreate(s, 2, meta->dtype, device, dev_id);
            model->s_x2 = tensorCreate(s, 2, meta->dtype, device, dev_id);
            model->s_o  = tensorCreate(s, 2, meta->dtype, device, dev_id);
        }

        // qflat: [maxseq, hs]
        {
            size_t s[2] = { MS, HS };
            model->s_qflat = tensorCreate(s, 2, meta->dtype, device, dev_id);
        }

        // kflat/vflat: [maxseq, kv]
        {
            size_t s[2] = { MS, KV };
            model->s_kflat = tensorCreate(s, 2, meta->dtype, device, dev_id);
            model->s_vflat = tensorCreate(s, 2, meta->dtype, device, dev_id);
        }

        // attn: [maxseq, nh, dh]
        {
            size_t s[3] = { MS, NH, DH };
            model->s_attn = tensorCreate(s, 3, meta->dtype, device, dev_id);

            // view as [maxseq, hs]
            size_t s2[2] = { MS, HS };
            model->s_attn_flat = tensorView(model->s_attn, s2, 2);
        }

        // o_proj bias: [hs] zeros
        {
            size_t s[1] = { HS };
            model->s_o_bias = tensorCreate(s, 1, meta->dtype, device, dev_id);
            std::memset(tensorGetData(model->s_o_bias), 0, HS * 2); // bf16 2 bytes
        }

        // last token scratch: [1, hs], [1, voc], argmax, lm bias
        {
            size_t s1[2] = { 1, HS };
            model->s_last      = tensorCreate(s1, 2, meta->dtype, device, dev_id);
            model->s_last_norm = tensorCreate(s1, 2, meta->dtype, device, dev_id);

            size_t slog[2] = { 1, VOC };
            model->s_logits = tensorCreate(slog, 2, meta->dtype, device, dev_id);

            size_t slog1d[1] = { VOC };
            model->s_logits_1d = tensorView(model->s_logits, slog1d, 1);

            size_t si[1] = { 1 };
            model->s_arg_i = tensorCreate(si, 1, LLAISYS_DTYPE_I64, device, dev_id);
            model->s_arg_v = tensorCreate(si, 1, meta->dtype, device, dev_id);

            size_t sb[1] = { VOC };
            model->s_lm_bias = tensorCreate(sb, 1, meta->dtype, device, dev_id);
            std::memset(tensorGetData(model->s_lm_bias), 0, VOC * 2);
        }

          // x2_norm: [MS, HS]
        {
            size_t s[2] = { MS, HS };
            model->s_x2_norm = tensorCreate(s, 2, meta->dtype, device, dev_id);
            model->s_x3      = tensorCreate(s, 2, meta->dtype, device, dev_id);
            model->s_down    = tensorCreate(s, 2, meta->dtype, device, dev_id);
        }

        // gate/up/act: [MS, DI]
        {
            size_t s[2] = { MS, DI };
            model->s_gate = tensorCreate(s, 2, meta->dtype, device, dev_id);
            model->s_up   = tensorCreate(s, 2, meta->dtype, device, dev_id);
            model->s_act  = tensorCreate(s, 2, meta->dtype, device, dev_id);
        }

        // biases: [DI] / [HS] zeros
        {
            size_t sdi[1] = { DI };
            model->s_gate_bias = tensorCreate(sdi, 1, meta->dtype, device, dev_id);
            model->s_up_bias   = tensorCreate(sdi, 1, meta->dtype, device, dev_id);
            std::memset(tensorGetData(model->s_gate_bias), 0, DI * 2);
            std::memset(tensorGetData(model->s_up_bias),   0, DI * 2);

            size_t shs[1] = { HS };
            model->s_down_bias = tensorCreate(shs, 1, meta->dtype, device, dev_id);
            std::memset(tensorGetData(model->s_down_bias), 0, HS * 2);
        }
    }
    return model;
}

__export void llaisysQwen2ModelDestroy(LlaisysQwen2Model* model) {
    if (!model) return;

    if (model->w.in_embed) tensorDestroy(model->w.in_embed);
    if (model->w.out_embed) tensorDestroy(model->w.out_embed);
    if (model->w.out_norm_w) tensorDestroy(model->w.out_norm_w);

    const size_t L = model->meta.nlayer;

    // per-layer: destroy each tensor
    for (size_t i = 0; i < L; ++i) {
        if (model->w.attn_norm_w && model->w.attn_norm_w[i]) tensorDestroy(model->w.attn_norm_w[i]);
        if (model->w.attn_q_w && model->w.attn_q_w[i]) tensorDestroy(model->w.attn_q_w[i]);
        if (model->w.attn_q_b && model->w.attn_q_b[i]) tensorDestroy(model->w.attn_q_b[i]);
        if (model->w.attn_k_w && model->w.attn_k_w[i]) tensorDestroy(model->w.attn_k_w[i]);
        if (model->w.attn_k_b && model->w.attn_k_b[i]) tensorDestroy(model->w.attn_k_b[i]);
        if (model->w.attn_v_w && model->w.attn_v_w[i]) tensorDestroy(model->w.attn_v_w[i]);
        if (model->w.attn_v_b && model->w.attn_v_b[i]) tensorDestroy(model->w.attn_v_b[i]);
        if (model->w.attn_o_w && model->w.attn_o_w[i]) tensorDestroy(model->w.attn_o_w[i]);

        if (model->w.mlp_norm_w && model->w.mlp_norm_w[i]) tensorDestroy(model->w.mlp_norm_w[i]);
        if (model->w.mlp_gate_w && model->w.mlp_gate_w[i]) tensorDestroy(model->w.mlp_gate_w[i]);
        if (model->w.mlp_up_w && model->w.mlp_up_w[i]) tensorDestroy(model->w.mlp_up_w[i]);
        if (model->w.mlp_down_w && model->w.mlp_down_w[i]) tensorDestroy(model->w.mlp_down_w[i]);
    }

    if (model->s_tokens) tensorDestroy(model->s_tokens);
    if (model->s_pos) tensorDestroy(model->s_pos);
    if (model->s_x0) tensorDestroy(model->s_x0);
    if (model->s_x1) tensorDestroy(model->s_x1);
    if (model->s_qflat) tensorDestroy(model->s_qflat);
    if (model->s_kflat) tensorDestroy(model->s_kflat);
    if (model->s_vflat) tensorDestroy(model->s_vflat);
    if (model->s_attn) tensorDestroy(model->s_attn);
    if (model->s_attn_flat) tensorDestroy(model->s_attn_flat);
    if (model->s_o) tensorDestroy(model->s_o);
    if (model->s_o_bias) tensorDestroy(model->s_o_bias);
    if (model->s_x2) tensorDestroy(model->s_x2);
    if (model->s_last) tensorDestroy(model->s_last);
    if (model->s_last_norm) tensorDestroy(model->s_last_norm);
    if (model->s_logits) tensorDestroy(model->s_logits);
    if (model->s_logits_1d) tensorDestroy(model->s_logits_1d);
    if (model->s_arg_i) tensorDestroy(model->s_arg_i);
    if (model->s_arg_v) tensorDestroy(model->s_arg_v);
    if (model->s_lm_bias) tensorDestroy(model->s_lm_bias);

    if (model->s_x2_norm) tensorDestroy(model->s_x2_norm);
    if (model->s_x3) tensorDestroy(model->s_x3);
    if (model->s_gate) tensorDestroy(model->s_gate);
    if (model->s_up) tensorDestroy(model->s_up);
    if (model->s_act) tensorDestroy(model->s_act);
    if (model->s_down) tensorDestroy(model->s_down);
    if (model->s_gate_bias) tensorDestroy(model->s_gate_bias);
    if (model->s_up_bias) tensorDestroy(model->s_up_bias);
    if (model->s_down_bias) tensorDestroy(model->s_down_bias);

    // free arrays
    if (model->w.attn_norm_w) std::free(model->w.attn_norm_w);
    if (model->w.attn_q_w) std::free(model->w.attn_q_w);
    if (model->w.attn_q_b) std::free(model->w.attn_q_b);
    if (model->w.attn_k_w) std::free(model->w.attn_k_w);
    if (model->w.attn_k_b) std::free(model->w.attn_k_b);
    if (model->w.attn_v_w) std::free(model->w.attn_v_w);
    if (model->w.attn_v_b) std::free(model->w.attn_v_b);
    if (model->w.attn_o_w) std::free(model->w.attn_o_w);

    if (model->w.mlp_norm_w) std::free(model->w.mlp_norm_w);
    if (model->w.mlp_gate_w) std::free(model->w.mlp_gate_w);
    if (model->w.mlp_up_w) std::free(model->w.mlp_up_w);
    if (model->w.mlp_down_w) std::free(model->w.mlp_down_w);

    if (model->cache_k) {
        for (size_t i = 0; i < model->meta.nlayer; ++i)
            tensorDestroy(model->cache_k[i]);
        std::free(model->cache_k);
    }

    if (model->cache_v) {
        for (size_t i = 0; i < model->meta.nlayer; ++i)
            tensorDestroy(model->cache_v[i]);
        std::free(model->cache_v);
    }

    std::free(model);
}

__export LlaisysQwen2Weights* llaisysQwen2ModelWeights(LlaisysQwen2Model* model){
    return &model->w;
}

__export int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model* model, int64_t* token_ids, size_t ntoken) {
    const size_t qlen = ntoken;
    const size_t HS = model->meta.hs;
    const size_t NH = model->meta.nh;
    const size_t DH = model->meta.dh;
    const size_t KVH = model->meta.nkvh;
    const float  scale = 1.0f / std::sqrt((float)DH);

    // ---------- basic guards ----------
    // ASSERT(qlen <= model->meta.maxseq, "infer: qlen > maxseq");
    // ASSERT(qlen >= 1, "infer: qlen must be >= 1");

    const bool is_prefill = (model->cur_pos == 0 && qlen > 1);
    
    static int dbg = 0;
    if (dbg < 50) {
        fprintf(stderr, "[infer] ntoken=%zu cur_pos=%zu prefill=%d\n",
                ntoken, model->cur_pos, (int)is_prefill);
        fflush(stderr);
        dbg++;
    }

    // ---------- write token ids into s_tokens[0:qlen] ----------
    {
        llaisysTensor_t tok_tmp = tensorSlice(model->s_tokens, 0, 0, qlen);
        tensorLoad(tok_tmp, token_ids);
        tensorDestroy(tok_tmp);
    }

    if (is_prefill) {
        // ============================================================
        // PREFILL: compute full [0:qlen), and fill cache [0:qlen)
        // ============================================================

        // tok_view: [qlen]
        llaisysTensor_t tok_view = tensorSlice(model->s_tokens, 0, 0, qlen);
        // x0_view: [qlen, hs]
        llaisysTensor_t x0_view  = tensorSlice(model->s_x0,     0, 0, qlen);

        llaisysEmbedding(x0_view, tok_view, model->w.in_embed);

        llaisysTensor_t cur_view = x0_view; // current hidden [qlen, hs]

        for (size_t layer = 0; layer < model->meta.nlayer; ++layer) {
            // ---- rmsnorm (attn) ----
            llaisysTensor_t x1_view = tensorSlice(model->s_x1, 0, 0, qlen);
            llaisysRmsNorm(x1_view, cur_view, model->w.attn_norm_w[layer], model->meta.epsilon);

            // ---- q/k/v linear (flat) ----
            llaisysTensor_t qflat = tensorSlice(model->s_qflat, 0, 0, qlen); // [qlen, hs]
            llaisysTensor_t kflat = tensorSlice(model->s_kflat, 0, 0, qlen); // [qlen, kv]
            llaisysTensor_t vflat = tensorSlice(model->s_vflat, 0, 0, qlen); // [qlen, kv]

            llaisysLinear(qflat, x1_view, model->w.attn_q_w[layer], model->w.attn_q_b[layer]);
            llaisysLinear(kflat, x1_view, model->w.attn_k_w[layer], model->w.attn_k_b[layer]);
            llaisysLinear(vflat, x1_view, model->w.attn_v_w[layer], model->w.attn_v_b[layer]);

            // ---- reshape to 3D ----
            size_t qshape[3] = { qlen, NH,  DH };
            size_t kvshape[3]= { qlen, KVH, DH };

            llaisysTensor_t q = tensorView(qflat, qshape, 3);
            llaisysTensor_t k = tensorView(kflat, kvshape, 3);
            llaisysTensor_t v = tensorView(vflat, kvshape, 3);

            // ---- RoPE ----
            llaisysTensor_t pos = tensorSlice(model->s_pos, 0, 0, qlen);
            llaisysROPE(q, q, pos, model->meta.theta);
            llaisysROPE(k, k, pos, model->meta.theta);

            // ---- write cache [0:qlen) ----
            {
                llaisysTensor_t ck = tensorSlice(model->cache_k[layer], 0, 0, qlen);
                llaisysTensor_t cv = tensorSlice(model->cache_v[layer], 0, 0, qlen);
                tensorLoad(ck, tensorGetData(k));
                tensorLoad(cv, tensorGetData(v));
                tensorDestroy(ck);
                tensorDestroy(cv);
            }

            // ---- attention (full prefill) ----
            llaisysTensor_t attn = tensorSlice(model->s_attn, 0, 0, qlen); // [qlen, nh, dh]
            llaisysSelfAttention(attn, q, k, v, scale);

            // ---- flatten attn -> [qlen, hs] ----
            size_t attn2shape[2] = { qlen, HS };
            llaisysTensor_t attn_flat = tensorView(attn, attn2shape, 2);

            // ---- o_proj + residual ----
            llaisysTensor_t o  = tensorSlice(model->s_o,  0, 0, qlen); // [qlen, hs]
            llaisysTensor_t x2 = tensorSlice(model->s_x2, 0, 0, qlen); // [qlen, hs]

            llaisysLinear(o, attn_flat, model->w.attn_o_w[layer], model->s_o_bias);
            llaisysAdd(x2, cur_view, o);

            // ---- MLP ----
            llaisysTensor_t x2n  = tensorSlice(model->s_x2_norm, 0, 0, qlen);
            llaisysTensor_t gate = tensorSlice(model->s_gate,    0, 0, qlen);
            llaisysTensor_t up   = tensorSlice(model->s_up,      0, 0, qlen);
            llaisysTensor_t act  = tensorSlice(model->s_act,     0, 0, qlen);
            llaisysTensor_t down = tensorSlice(model->s_down,    0, 0, qlen);
            llaisysTensor_t x3   = tensorSlice(model->s_x3,      0, 0, qlen);

            llaisysRmsNorm(x2n, x2, model->w.mlp_norm_w[layer], model->meta.epsilon);
            llaisysLinear(gate, x2n, model->w.mlp_gate_w[layer], model->s_gate_bias);
            llaisysLinear(up,   x2n, model->w.mlp_up_w[layer],   model->s_up_bias);
            llaisysSwiGLU(act, gate, up);
            llaisysLinear(down, act, model->w.mlp_down_w[layer], model->s_down_bias);
            llaisysAdd(x3, x2, down);

            // next layer input
            if (cur_view != x0_view) tensorDestroy(cur_view);
            cur_view = x3;

            // ---- destroy per-layer temporaries (keep cur_view) ----
            tensorDestroy(x1_view);
            tensorDestroy(qflat);
            tensorDestroy(kflat);
            tensorDestroy(vflat);
            tensorDestroy(q);
            tensorDestroy(k);
            tensorDestroy(v);
            tensorDestroy(pos);
            tensorDestroy(attn);
            tensorDestroy(attn_flat);
            tensorDestroy(o);
            tensorDestroy(x2);
            tensorDestroy(x2n);
            tensorDestroy(gate);
            tensorDestroy(up);
            tensorDestroy(act);
            tensorDestroy(down);
            // x3 kept as cur_view
        }

        // ---- last token logits ----
        llaisysTensor_t last = tensorSlice(cur_view, 0, qlen - 1, qlen); // [1, hs]
        llaisysRmsNorm(model->s_last_norm, last, model->w.out_norm_w, model->meta.epsilon);
        llaisysLinear(model->s_logits, model->s_last_norm, model->w.out_embed, model->s_lm_bias);
        llaisysArgmax(model->s_arg_i, model->s_arg_v, model->s_logits_1d);
        int64_t out_id = *reinterpret_cast<int64_t*>(tensorGetData(model->s_arg_i));

        // cleanup
        tensorDestroy(last);
        if (cur_view != x0_view) tensorDestroy(cur_view);
        tensorDestroy(x0_view);
        tensorDestroy(tok_view);

        model->cur_pos = qlen;
        return out_id;
    }

    // ============================================================
    // DECODE: only compute last token t=qlen-1 using KV cache
    // ============================================================

    const size_t t = qlen - 1;
    const size_t cached = model->cur_pos;
    ASSERT(cached == t, "decode: cached != t (sequence mismatch)");

    // tok_last: [1]
    llaisysTensor_t tok_last = tensorSlice(model->s_tokens, 0, t, t + 1);
    // x0_last: [1, hs]  (写在 s_x0 的最后一行也行，但你现在用 slice，就沿用)
    llaisysTensor_t x0_last  = tensorSlice(model->s_x0,     0, t, t + 1);
    llaisysEmbedding(x0_last, tok_last, model->w.in_embed);

    llaisysTensor_t cur_last = x0_last; // [1, hs]

    for (size_t layer = 0; layer < model->meta.nlayer; ++layer) {
        // rmsnorm
        llaisysTensor_t x1_1 = tensorSlice(model->s_x1, 0, 0, 1);
        llaisysRmsNorm(x1_1, cur_last, model->w.attn_norm_w[layer], model->meta.epsilon);

        // q/k/v flat for 1 token
        llaisysTensor_t qflat_1 = tensorSlice(model->s_qflat, 0, 0, 1);
        llaisysTensor_t kflat_1 = tensorSlice(model->s_kflat, 0, 0, 1);
        llaisysTensor_t vflat_1 = tensorSlice(model->s_vflat, 0, 0, 1);

        llaisysLinear(qflat_1, x1_1, model->w.attn_q_w[layer], model->w.attn_q_b[layer]);
        llaisysLinear(kflat_1, x1_1, model->w.attn_k_w[layer], model->w.attn_k_b[layer]);
        llaisysLinear(vflat_1, x1_1, model->w.attn_v_w[layer], model->w.attn_v_b[layer]);

        // reshape to 3D
        size_t qshape1[3]  = { 1, NH,  DH };
        size_t kvshape1[3] = { 1, KVH, DH };
        llaisysTensor_t q1 = tensorView(qflat_1, qshape1, 3);
        llaisysTensor_t k1 = tensorView(kflat_1, kvshape1, 3);
        llaisysTensor_t v1 = tensorView(vflat_1, kvshape1, 3);

        // RoPE for this position
        llaisysTensor_t pos1 = tensorSlice(model->s_pos, 0, t, t + 1);
        llaisysROPE(q1, q1, pos1, model->meta.theta);
        llaisysROPE(k1, k1, pos1, model->meta.theta);

        // write cache at [t:t+1]
        {
            llaisysTensor_t ck = tensorSlice(model->cache_k[layer], 0, t, t + 1);
            llaisysTensor_t cv = tensorSlice(model->cache_v[layer], 0, t, t + 1);
            tensorLoad(ck, tensorGetData(k1));
            tensorLoad(cv, tensorGetData(v1));
            tensorDestroy(ck);
            tensorDestroy(cv);
        }

        // attention context uses [0:t+1]
        llaisysTensor_t kctx = tensorSlice(model->cache_k[layer], 0, 0, t + 1);
        llaisysTensor_t vctx = tensorSlice(model->cache_v[layer], 0, 0, t + 1);

        // attn out [1, nh, dh]
        llaisysTensor_t attn1 = tensorSlice(model->s_attn, 0, 0, 1);
        llaisysSelfAttention(attn1, q1, kctx, vctx, scale);

        // flatten [1, hs]
        size_t attnflat1[2] = { 1, HS };
        llaisysTensor_t attn_flat_1 = tensorView(attn1, attnflat1, 2);

        // o_proj + residual
        llaisysTensor_t o1  = tensorSlice(model->s_o,  0, 0, 1);
        llaisysTensor_t x2_1= tensorSlice(model->s_x2, 0, 0, 1);
        llaisysLinear(o1, attn_flat_1, model->w.attn_o_w[layer], model->s_o_bias);
        llaisysAdd(x2_1, cur_last, o1);

        // MLP
        llaisysTensor_t x2n_1 = tensorSlice(model->s_x2_norm, 0, 0, 1);
        llaisysTensor_t gate1 = tensorSlice(model->s_gate,    0, 0, 1);
        llaisysTensor_t up1   = tensorSlice(model->s_up,      0, 0, 1);
        llaisysTensor_t act1  = tensorSlice(model->s_act,     0, 0, 1);
        llaisysTensor_t down1 = tensorSlice(model->s_down,    0, 0, 1);
        llaisysTensor_t x3_1  = tensorSlice(model->s_x3,      0, 0, 1);

        llaisysRmsNorm(x2n_1, x2_1, model->w.mlp_norm_w[layer], model->meta.epsilon);
        llaisysLinear(gate1, x2n_1, model->w.mlp_gate_w[layer], model->s_gate_bias);
        llaisysLinear(up1,   x2n_1, model->w.mlp_up_w[layer],   model->s_up_bias);
        llaisysSwiGLU(act1, gate1, up1);
        llaisysLinear(down1, act1, model->w.mlp_down_w[layer], model->s_down_bias);
        llaisysAdd(x3_1, x2_1, down1);

        // next layer input
        if (cur_last != x0_last) tensorDestroy(cur_last);
        cur_last = x3_1;

        // cleanup
        tensorDestroy(x1_1);
        tensorDestroy(qflat_1);
        tensorDestroy(kflat_1);
        tensorDestroy(vflat_1);
        tensorDestroy(q1);
        tensorDestroy(k1);
        tensorDestroy(v1);
        tensorDestroy(pos1);
        tensorDestroy(kctx);
        tensorDestroy(vctx);
        tensorDestroy(attn1);
        tensorDestroy(attn_flat_1);
        tensorDestroy(o1);
        tensorDestroy(x2_1);
        tensorDestroy(x2n_1);
        tensorDestroy(gate1);
        tensorDestroy(up1);
        tensorDestroy(act1);
        tensorDestroy(down1);
        // x3_1 kept as cur_last
    }

    // logits from cur_last ([1,hs])
    llaisysRmsNorm(model->s_last_norm, cur_last, model->w.out_norm_w, model->meta.epsilon);
    llaisysLinear(model->s_logits, model->s_last_norm, model->w.out_embed, model->s_lm_bias);
    llaisysArgmax(model->s_arg_i, model->s_arg_v, model->s_logits_1d);
    int64_t out_id = *reinterpret_cast<int64_t*>(tensorGetData(model->s_arg_i));

    // update cache pos
    model->cur_pos = qlen;

    // cleanup
    if (cur_last != x0_last) tensorDestroy(cur_last);
    tensorDestroy(x0_last);
    tensorDestroy(tok_last);

    return out_id;
}
}
