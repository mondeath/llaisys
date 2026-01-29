#include "op.hpp"
#include<cmath>


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

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    ASSERT(in->ndim() == 3,  "ops::rope: input must be 3D");
    ASSERT(out->ndim() == 3, "ops::rope: output must be 3D");
    ASSERT(pos_ids->ndim() == 1, "ops::rope: pos_ids must be 1D");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "ops::rope: pos_ids dtype must be int64");

    int64_t S = static_cast<int64_t>(in->shape()[0]);
    int64_t H = static_cast<int64_t>(in->shape()[1]);
    int64_t D = static_cast<int64_t>(in->shape()[2]);

    ASSERT(out->shape()[0] == static_cast<size_t>(S), "ops::rope: S mismatch");
    ASSERT(out->shape()[1] == static_cast<size_t>(H), "ops::rope: H mismatch");
    ASSERT(out->shape()[2] == static_cast<size_t>(D), "ops::rope: D mismatch");

    ASSERT(static_cast<int64_t>(pos_ids->shape()[0]) == S, "ops::rope: pos_ids length mismatch");//pos表示token的位置
    ASSERT((D % 2) == 0, "ops::rope: D must be even");

    const void* x = in->data();
    void* y = out->data();
    const int64_t* p = reinterpret_cast<const int64_t *>(pos_ids->data());
    
    llaisysDataType_t dt = in->dtype();
    ASSERT(out->dtype() == dt, "ops::rope: out dtype mismatch");

    int64_t half = D / 2;

    std::vector<float> denom(half);
    for(int64_t j = 0; j < half; j++){
        float exponent = static_cast<float>(2.0f * j) / static_cast<float>(D);
        denom[j] = std::pow(theta, exponent); 
    }

    switch(in->deviceType())
    {
    case LLAISYS_DEVICE_CPU:{
        for(int64_t s = 0; s < S; s++){
            float ps = static_cast<float>(p[s]);
            for(int64_t h = 0; h < H; h++){
                int64_t base = (s * H + h) * D;

                for(int64_t i = 0; i < half; i++){
                    float phi = ps / denom[i];
                    float co = std::cos(phi);
                    float si = std::sin(phi);

                    int64_t a_idx = base + i;
                    int64_t b_idx = base + i + half;

                    float a = load_as_f32(x, a_idx, dt);
                    float b = load_as_f32(x, b_idx, dt);

                    float a2 = a * co - b * si;
                    float b2 = b * co + a * si;

                    write_back(y, a_idx, dt, a2);
                    write_back(y, b_idx, dt, b2);
                }
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
