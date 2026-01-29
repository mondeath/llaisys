#include "op.hpp"
#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {//“整行拷贝”，out 的每一行，完全等于 weight 中被 index 指到的那一行
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "ops::embedding: index dtype must be i64");   

    const size_t N = index->shape()[0];
    const size_t D = weight->shape()[1];

    ASSERT(out->shape()[0] == N, "ops::embedding: outshape[0] dismatch");
    ASSERT(out->shape()[1] == D,"ops::embedding: outshape[1] distmatch");
    ASSERT(out->dtype() == weight->dtype(), "ops::embedding: out dtype must match the weight dtype");

    const auto dtype = weight->dtype();    
    const size_t row_bytes = (size_t) D * llaisys::utils::dsize(dtype);//一行跨度

    const int64_t* idx = reinterpret_cast<const int64_t*>(index->data());//data()为byte*，需要强制转换
    std::byte* out_p = out->data();
    const std::byte* weight_p = weight->data();

    const int64_t V = static_cast<int64_t>(weight->shape()[0]);
    for(int64_t i = 0; i < static_cast<int64_t>(N); i++){
        const int64_t r = idx[i];
        ASSERT(0 <= r && r < V, "ops::embedding: index out of range");

        std::memcpy(
            out_p + (size_t) i * row_bytes,
            weight_p + (size_t) r * row_bytes,
            row_bytes
        );
    }
}
} // namespace llaisys::ops
