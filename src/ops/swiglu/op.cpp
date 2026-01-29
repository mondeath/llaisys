#include "op.hpp"

#include <cmath>

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

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    ASSERT(out->ndim() == 2,  "ops::swiglu: out must be 2D");
    ASSERT(gate->ndim() == 2, "ops::swiglu: gate must be 2D");
    ASSERT(up->ndim() == 2,   "ops::swiglu: up must be 2D");

    ASSERT(out->shape()[0] == gate->shape()[0] && out->shape()[1] == gate->shape()[1],
           "ops::swiglu: out/gate shape mismatch");
    ASSERT(out->shape()[0] == up->shape()[0] && out->shape()[1] == up->shape()[1],
           "ops::swiglu: out/up shape mismatch");

    ASSERT(out->dtype() == gate->dtype() && out->dtype() == up->dtype(),
           "ops::swiglu: dtype mismatch");

    const auto dt = out->dtype();
    const int64_t n0 = out->shape()[0];
    const int64_t n1 = out->shape()[1];
    const int64_t numel = n0 * n1;

    const void* gate_p = gate->data();
    const void* up_p = up->data();
    void* out_p = out->data();

    for(int64_t i = 0; i < numel; i++){
        float g = load_as_f32(gate_p, i, dt);
        float u = load_as_f32(up_p, i, dt);

        float denom = 1.0f + std::exp(-g);
        float y = u * (g / denom);

        write_back(out_p, i, dt, y);
    }

}
} // namespace llaisys::ops
