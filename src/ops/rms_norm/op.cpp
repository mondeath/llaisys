#include "op.hpp"
#include<cmath>

namespace llaisys::ops {
static inline float load_as_f32(const void* base, int64_t idx, llaisysDataType_t dt) {
    switch (dt) {
        case LLAISYS_DTYPE_F32:
            return llaisys::utils::cast<float>(reinterpret_cast<const float*>(base)[idx]);
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
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    ASSERT(in->ndim() == 2, "ops::rms_norm: only supports 2D input tensors.");
    ASSERT(out->ndim() == 2, "ops::rms_norm: only supports 2D output tensors.");
    ASSERT(weight->ndim() == 1, "ops::rms_norm: weight must be 1D");

    int64_t N = static_cast<int64_t>(in->shape()[0]);
    int64_t d = static_cast<int64_t>(in->shape()[1]);

    auto dt = in->dtype();
    ASSERT(weight->dtype() == dt, "ops::rms_norm: weight dtype mismatch");
    ASSERT(out->dtype() == dt, "ops::rms_norm: out dtype mismatch");

    const void* x = in->data();
    const void* w = weight->data();
    void* y = out->data();

    switch(in->deviceType())
    {
    case LLAISYS_DEVICE_CPU:{
        for(int64_t n = 0; n < N; n++){
            int64_t base = n * d;

            float sum_sq = 0.f;
            for(int64_t k = 0; k < d; k++){
                float x_v = load_as_f32(x, base + k, dt);
                sum_sq += x_v * x_v;
            }   

            float mean_sq = sum_sq / static_cast<float>(d);
            float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

            for(int64_t k = 0; k < d; k++){
                float x_v = load_as_f32(x, base + k, dt);
                float w_v = load_as_f32(w, k, dt);
                float out_v = x_v * inv_rms * w_v;
                write_back(y, base + k, dt, out_v);
            }
        }
    break;
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
