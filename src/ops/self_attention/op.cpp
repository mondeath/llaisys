#include "op.hpp"

#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>

namespace llaisys::ops {
static inline float load_as_f32(const void* base, int64_t idx, llaisysDataType_t dt) {
    switch (dt) {
        case LLAISYS_DTYPE_F32:
            return reinterpret_cast<const float*>(base)[idx];
        case LLAISYS_DTYPE_F16:
            return llaisys::utils::cast<float>(reinterpret_cast<const fp16_t*>(base)[idx]);
        case LLAISYS_DTYPE_BF16:
            return llaisys::utils::cast<float>(reinterpret_cast<const bf16_t*>(base)[idx]);
        default:
            ASSERT(false, "unsupported dtype");
            return 0.f;
    }
}

static inline void write_back(void* base, int64_t idx, llaisysDataType_t dt, float v){
    switch (dt)
    {
    case LLAISYS_DTYPE_F32:
        reinterpret_cast<float*>(base)[idx] = v;
        break;
    case LLAISYS_DTYPE_F16:
        reinterpret_cast<fp16_t*>(base)[idx] = llaisys::utils::cast<fp16_t>(v);
        break;
    case LLAISYS_DTYPE_BF16:
        reinterpret_cast<bf16_t*>(base)[idx] = llaisys::utils::cast<bf16_t>(v);
        break;
    default:
        ASSERT(false, "unsupported dtype");
        break;
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    ASSERT(q->ndim() == 3, "self_attention: q must be [seqlen, nhead, d]");
    ASSERT(k->ndim() == 3, "self_attention: k must be [total_len, nkvhead, d]");
    ASSERT(v->ndim() == 3, "self_attention: v must be [total_len, nkvhead, dv]");
    ASSERT(attn_val->ndim() == 3, "self_attention: attn_val must be [seqlen, nhead, dv]");

    const int64_t seqlen  = static_cast<int64_t>(q->shape()[0]);
    const int64_t nhead   = static_cast<int64_t>(q->shape()[1]);
    const int64_t d       = static_cast<int64_t>(q->shape()[2]);

    const int64_t total_len = static_cast<int64_t>(k->shape()[0]);
    const int64_t nkvhead   = static_cast<int64_t>(k->shape()[1]);
    const int64_t kd        = static_cast<int64_t>(k->shape()[2]);

    const int64_t vd_total_len = static_cast<int64_t>(v->shape()[0]);
    const int64_t vd_nkvhead   = static_cast<int64_t>(v->shape()[1]);
    const int64_t dv           = static_cast<int64_t>(v->shape()[2]);

    ASSERT(kd == d, "self_attention: k last dim must equal q last dim (d)");
    ASSERT(vd_total_len == total_len, "self_attention: v total_len must match k total_len");
    ASSERT(vd_nkvhead == nkvhead, "self_attention: v nkvhead must match k nkvhe");

    ASSERT(static_cast<int64_t>(attn_val->shape()[0]) == seqlen, "self_attention: out seqlen mismatch");
    ASSERT(static_cast<int64_t>(attn_val->shape()[1]) == nhead, "self_attention: out nhead mismatch");
    ASSERT(static_cast<int64_t>(attn_val->shape()[2]) == dv, "self_attention: out dv mismatch");

    const int64_t past_len = total_len - seqlen;
    ASSERT(past_len >= 0, "self_attention: total_len must be >= seqlen");

    const void* q_base = q->data();
    const void* k_base = k->data();
    const void* v_base = v->data();
    void* out_base = attn_val->data();

    const auto q_dt = q->dtype();
    const auto k_dt = k->dtype();
    const auto v_dt = v->dtype();
    const auto o_dt = attn_val->dtype();

    std::vector<float> logits;   // softmax 前分数
    std::vector<float> probs;    // softmax 后权重

    for(int64_t t = 0; t < seqlen; t++){
        const int64_t max_j = past_len + t;//此前所有token
        const int64_t L = max_j + 1;//当前所有

        logits.resize(static_cast<size_t>(L));
        probs.resize(static_cast<size_t>(L));

        for(int64_t h = 0; h < nhead; h++){
            ASSERT(nkvhead > 0, "self_attention: nkvhead must be > 0");
            ASSERT(nhead % nkvhead == 0, "self_attention: nhead must be divisible by nkvhead for GQA");

            const int64_t group = nhead / nkvhead;   // 每个 kv head 对应多少个 q head
            const int64_t kvh = h / group;           // 连续分组映射（Torch 风格）

            float m = -std::numeric_limits<float>::infinity();

            for(int64_t j = 0; j < L; j++){//计算max 并存储其他logits值
                float acc = 0.0f;
                const int64_t q_row = (t * nhead + h) * d;
                const int64_t k_row = (j * nkvhead + kvh) * d;
                
                for(int64_t i = 0; i < d; i++){
                    const float qv = load_as_f32(q_base, q_row + i, q_dt);
                    const float kv = load_as_f32(k_base, k_row + i, k_dt);
                    acc += qv * kv;
                }

                const float s = acc * scale;
                logits[static_cast<size_t>(j)] = s;
                if(s > m) m = s;
            }

            float sum = 0.0f;
            for(int64_t j = 0; j < L; j++){//e^指数
                const float e = std::exp(logits[static_cast<size_t>(j)] - m);
                probs[static_cast<size_t>(j)] = e;
                sum += e;
            }

            const int64_t out_row = (t * nhead + h) * dv;

            std::vector<float> out_acc;
            out_acc.assign(static_cast<size_t>(dv), 0.0f);

  
            const float inv_sum = 1.0f / sum;

            for (int64_t j = 0; j < L; ++j) {
                const float w = probs[static_cast<size_t>(j)] * inv_sum;
                const int64_t v_row = (j * nkvhead + kvh) * dv;

                for (int64_t u = 0; u < dv; ++u) {
                    const float vv = load_as_f32(v_base, v_row + u, v_dt);
                    out_acc[static_cast<size_t>(u)] += w * vv; // 始终 float 累加
                }
            }

            for (int64_t u = 0; u < dv; ++u) { //最后一次性写回输出 dtype（f16/bf16/f32）
                write_back(out_base, out_row + u, o_dt, out_acc[static_cast<size_t>(u)]);
            }
        }
    }
}
} // namespace llaisys::ops
