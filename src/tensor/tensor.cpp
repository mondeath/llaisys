#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    if(numel() == 0) return true;

    const auto& strides = this->strides();
    const auto& shape = this->shape();
    ptrdiff_t expeceted_stride = 1;

    for(int i = strides.size() - 1; i >= 0; i--)
    {
        if(strides[i] != expeceted_stride){
            return false;
        }

        expeceted_stride *= shape[i];
    }
    
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const size_t n = ndim();
    if(order.size() != n){
        throw std::runtime_error("Tensor::permute: order size dismatch");
    }

    std::vector<bool> beseen(n, false);
    for(size_t i = 0; i < n; i++){
        if(order[i] > n){
            throw std::runtime_error("Tensor::permute: order size out of range");
        }
        if(beseen[order[i]]){
            throw std::runtime_error("Tensor::permute: duplicate dimension in order");
        }

        beseen[order[i]] = true;
    }

    std::vector<size_t> new_shape(n);
    std::vector<ptrdiff_t> new_strides(n);

    for(size_t i = 0; i < n; i++){
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }

    TensorMeta meta = {dtype(), new_shape, new_strides};
    return tensor_t(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &new_shape) const {
    auto prod = [](const std::vector<size_t>& v) -> size_t{
        size_t r = 1;
        for(size_t x: v) r *= x;
        return r;
    };

    if(prod(new_shape) != numel()){
        throw std::runtime_error("Tensor::view numel donesn't match");
    }

    if(numel() == 0){
        std::vector<ptrdiff_t> new_strides(new_shape.size(), 1);
        ptrdiff_t s = 1;
        for(size_t i = 1; i <= new_shape.size(); i++){
            new_strides[new_shape.size() - i] = s;
            s *= (ptrdiff_t)new_shape[new_shape.size() - i];
        }
        TensorMeta meta{dtype(), new_shape, new_strides};
        return tensor_t(new Tensor(meta, _storage, _offset));
    }

    //compress 
    std::vector<size_t> oshape;
    std::vector<ptrdiff_t> ostrides;
    oshape.reserve(_meta.shape.size());
    ostrides.reserve(_meta.strides.size());
    for(size_t i = 0; i < _meta.shape.size(); i++){
        if(_meta.shape[i] == 1) continue;
        oshape.push_back(_meta.shape[i]);
        ostrides.push_back(_meta.strides[i]);
    }

    std::vector<size_t> nshape_comp;
    nshape_comp.reserve(new_shape.size());
    for (size_t x : new_shape) {
        if (x == 1) continue;
        nshape_comp.push_back(x);
    }
    
    if(nshape_comp.empty()){
        std::vector<ptrdiff_t> new_strides(new_shape.size(), 1);
        TensorMeta meta{dtype(), new_shape, new_strides};
        return tensor_t(new Tensor(meta, _storage, _offset));
    }

    std::vector<ptrdiff_t> nstrides_comp(nshape_comp.size(), 0);

    size_t i = 0;
    size_t k = 0;

    while (i < oshape.size() && k < nshape_comp.size()) {
        size_t j = i;
        size_t t = k;

        size_t old_num = oshape[i];
        size_t new_num = nshape_comp[k];

        // 谁小扩谁，直到 old_num == new_num
        while (old_num != new_num) {
            if (old_num < new_num) {
                if (j + 1 >= oshape.size()) {
                    throw std::runtime_error("Tensor::view: incompatible shape");
                }
                // old 扩维必须保证可合并（无洞）
                if (ostrides[j] != ostrides[j + 1] * (ptrdiff_t)oshape[j + 1]) {
                    throw std::runtime_error("Tensor::view: incompatible shape -- not contiguous");
                }
                ++j;
                old_num *= oshape[j];
            } else { // old_num > new_num
                if (t + 1 >= nshape_comp.size()) {
                    throw std::runtime_error("Tensor::view: incompatible shape");
                }
                ++t;
                new_num *= nshape_comp[t];
            }
        }

        // old[i..j] 对应 new[k..t]
        ptrdiff_t base = ostrides[j];
        nstrides_comp[t] = base;

        for (size_t p = t; p-- > k; ) {
            nstrides_comp[p] = nstrides_comp[p + 1] * (ptrdiff_t)nshape_comp[p + 1];
        }

        i = j + 1;
        k = t + 1;
    }
        
    if (!(i == oshape.size() && k == nshape_comp.size())) {
        throw std::runtime_error("Tensor::view: incompatible shape");
    }

    std::vector<ptrdiff_t> new_strides(new_shape.size(), 1);
    size_t comp_idx = 0;
    for(size_t idx = 0; idx < new_shape.size(); idx++){
        if(new_shape[idx] == 1){
            new_strides[idx] = 1;
        }else{
            new_strides[idx] = nstrides_comp[comp_idx++];
        }
    }

    TensorMeta meta = {dtype(), new_shape, new_strides};
    return tensor_t (new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if(dim >= ndim()){
        throw std::runtime_error("Tensor::slice: dim out of range");
    }
    if(start > end){
        throw std::runtime_error("Tensor::slice: start > end");
    }
    if(end > _meta.shape[dim]){
        throw std::runtime_error("Tensor::slice: end out of range");
    }

    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;

    std::vector<ptrdiff_t> new_strides = _meta.strides;

    size_t new_offset = _offset + start * _meta.strides[dim] * elementSize();

    TensorMeta meta = {dtype(), new_shape, new_strides};
    return tensor_t(new Tensor(meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    if(!src_){
        throw std::invalid_argument("Tensor::load: src is null");
    }

    if(!isContiguous()){
        throw std::invalid_argument("Tensor::load: only contiguous tensor is supported");
    }

    const size_t bytes = numel() * elementSize();
    if(bytes == 0) return;

    const auto dev_type = deviceType();//llaisysDeviceType_t
    const int dev_id = deviceId();

    llaisysSetContextRuntime(dev_type, dev_id);

    const LlaisysRuntimeAPI *api = llaisysGetRuntimeAPI(dev_type);
    if(!api || !api->memcpy_sync){
        throw std::runtime_error("Tensor::load :runtime memcpu_sync error");
    }

    void *dst = static_cast<void *> (data());
    llaisysMemcpyKind_t kind = (dev_type == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;

    api->memcpy_sync(dst, src_, bytes, kind);

}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
