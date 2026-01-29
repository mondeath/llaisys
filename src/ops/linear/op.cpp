#include "op.hpp"

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
            ASSERT(false, "ops::linear: unsupported dtype");
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
        ASSERT(false, "ops::linear: unsupported dtype");
        break;
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    ASSERT(in->ndim() == 2, "ops::linear: input must be 2D");
    ASSERT(weight->ndim() == 2, "ops::linear: weigth must be 2D");
    ASSERT(out->ndim() == 2, "ops::linear: out must be 2D");

    int64_t N = static_cast<int64_t> (in->shape()[0]);
    int64_t K = static_cast<int64_t> (in->shape()[1]);
    int64_t M = static_cast<int64_t> (weight->shape()[0]);

    ASSERT(static_cast<int64_t>(weight->shape()[1]) == K, "ops::linear: K dismatch");
    ASSERT(static_cast<int64_t>(out->shape()[0]) == N, "ops::linear: N dismatch");
    ASSERT(static_cast<int64_t>(out->shape()[1]) == M, "ops::linear: M dismatch");    

    bool has_bias = (bool)bias;
    if (has_bias) {
        ASSERT(bias->ndim() == 1, "linear: bias must be 1D");
        ASSERT(static_cast<int64_t>(bias->shape()[0]) == M, "linear: bias dim mismatch");
    }

    auto dt = in->dtype();
    ASSERT(weight->dtype() == dt, "ops::linear: weight dtype mismatch");
    ASSERT(out->dtype() == dt, "ops::linear: out dtype mismatch");
    if (has_bias) ASSERT(bias->dtype() == dt, "ops::linear: bias dtype mismatch");

    ASSERT(dt == LLAISYS_DTYPE_F32 || dt == LLAISYS_DTYPE_F16 || dt == LLAISYS_DTYPE_BF16,
        "ops::linear: unsupported dtype");
    
    const void* x = in->data();
    const void* w = weight->data();
    const void* b = has_bias ? bias->data() : nullptr;
    void* y = out->data();
 
    switch (in->deviceType())
    {
    case LLAISYS_DEVICE_CPU:{
        for(int64_t n = 0; n < N; n++){
            for(int64_t m = 0; m < M; m++){
                float acc = 0.f;

                int64_t x_base = n * K;
                int64_t w_base = m * K;

                for(int64_t k = 0; k < K; k++){
                    float x_v = load_as_f32(x, x_base + k, dt);
                    float w_v = load_as_f32(w, w_base + k, dt);

                    acc += x_v * w_v;
                }

                if(has_bias){
                    acc += load_as_f32(b, m, dt);
                }

                write_back(y, n * M + m, dt, acc);
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
