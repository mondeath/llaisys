#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include<cmath>

namespace llaisys::ops {
    static inline void write_i64_index(void* out, uint64_t idx) {
        *reinterpret_cast<int64_t*>(out) = static_cast<int64_t>(idx);
    }

    static inline bool better(float v, float best) {//避免NAN情况的干扰 
        return (std::isnan(v) && !std::isnan(best)) || (v > best);
    }

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // ---- shape constraints (match test/ops/argmax.py) ----
    ASSERT(vals->shape().size() == 1, "ops::Argmax: vals must be 1D.");

    // max_idx and max_val are 1D tensors with a single element: shape (1,)
    ASSERT(max_idx->shape().size() == 1 && max_idx->numel() == 1,
        "ops::Argmax: max_idx must be a 1D tensor with 1 element.");
    ASSERT(max_val->shape().size() == 1 && max_val->numel() == 1,
        "ops::Argmax: max_val must be a 1D tensor with 1 element.");
        
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "ops::Argmax: max_idx dtype must be i64.");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "ops::Argmax: Add: all tensors must be contiguous.");

    const size_t n = vals->numel();
    ASSERT(n > 0, "ops::Argmax: vals must have at least one element.");

    switch (vals->deviceType()){
    case LLAISYS_DEVICE_CPU:{
        void* out_i = max_idx->data();
        void* out_v = max_val->data();
        auto dtype = vals->dtype();

        uint64_t best_i = 0;

        if(dtype == LLAISYS_DTYPE_F32){
            const float* x = reinterpret_cast<const float*>(vals->data());
            float best_v = x[0];
            for(uint64_t i = 1; i < n; i++){
                float v = x[i];
                if(better(v, best_v)){
                    best_v = v;
                    best_i = i;
                }
            }
            write_i64_index(out_i, best_i);//wirte back
            *reinterpret_cast<float*>(out_v) = x[best_i];
            return;
        } else if (dtype == LLAISYS_DTYPE_F16) {
            const uint16_t* x = reinterpret_cast<const uint16_t*>(vals->data());

            float best_v = llaisys::utils::cast<float>(llaisys::fp16_t{ x[0] });

            for (uint64_t i = 1; i < n; i++) {
                float v = llaisys::utils::cast<float>(llaisys::fp16_t{ x[i] });
                if (better(v, best_v)) {
                    best_v = v;
                    best_i = i;
                }
            }
        write_i64_index(out_i, best_i);
        *reinterpret_cast<uint16_t*>(out_v) = x[best_i];  
        return;
        } else if (dtype == LLAISYS_DTYPE_BF16) {
            const uint16_t* x = reinterpret_cast<const uint16_t*>(vals->data());

            float best_v = llaisys::utils::cast<float>(llaisys::bf16_t{ x[0] });

            for (uint64_t i = 1; i < n; i++) {
                float v = llaisys::utils::cast<float>(llaisys::bf16_t{ x[i] });
                if (better(v, best_v)) {
                    best_v = v;
                    best_i = i;
                }
            }

            write_i64_index(out_i, best_i);
            *reinterpret_cast<uint16_t*>(out_v) = x[best_i];  
            return;
        }
    }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
