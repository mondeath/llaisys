from typing import Sequence
import ctypes
from ctypes import c_int64, c_int

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta 

from pathlib import Path
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        import json

        model_path = Path(model_path)
        cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))

        meta = LlaisysQwen2Meta()
        meta.dtype = int(DataType.BF16) 
        meta.nlayer = int(cfg["num_hidden_layers"])
        meta.hs = int(cfg["hidden_size"])
        meta.nh = int(cfg["num_attention_heads"])
        meta.nkvh = int(cfg["num_key_value_heads"])
        meta.di = int(cfg["intermediate_size"])
        meta.voc = int(cfg["vocab_size"])
        meta.epsilon = float(cfg.get("rms_norm_eps", 1e-6))
        meta.theta = float(cfg.get("rope_theta", 10000))
        meta.dh = meta.hs // meta.nh
        meta.maxseq = int(cfg.get("sliding_window", 4096))

        # 停止 token
        eos = cfg.get("eos_token_id")
        meta.end_token = int(eos[0] if isinstance(eos, list) else eos)
        
        
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            int(device),
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate returned NULL")

        w_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not w_ptr:
            raise RuntimeError("llaisysQwen2ModelWeights returned NULL (backend not implemented yet)")
        self._w = w_ptr.contents
        self._meta = meta
        self._device = device

        model_path = Path(model_path)
        
        import torch 
        import re

        layer_re = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

        def _ptr(t):
            t = t.contiguous()
            return ctypes.c_void_p(t.data_ptr())
        
        loaded = 0
        skipped = 0

        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as data_:
                for name_ in data_.keys():
                    t = data_.get_tensor(name_)
                    
                    #global
                    if name_ == "model.embed_tokens.weight":
                        LIB_LLAISYS.tensorLoad(self._w.in_embed, _ptr(t)); loaded += 1
                        continue
                    elif name_ == "lm_head.weight":
                        LIB_LLAISYS.tensorLoad(self._w.out_embed, _ptr(t)); loaded += 1
                        continue
                    elif name_ == "model.norm.weight":
                        LIB_LLAISYS.tensorLoad(self._w.out_norm_w, _ptr(t)); loaded += 1
                        continue
                    
                    # --- per-layer ---
                    m = layer_re.match(name_)
                    if not m:
                        skipped += 1
                        continue

                    i = int(m.group(1))
                    suffix = m.group(2)

                    # norms
                    if suffix == "input_layernorm.weight":
                        LIB_LLAISYS.tensorLoad(self._w.attn_norm_w[i], _ptr(t)); loaded += 1
                    elif suffix == "post_attention_layernorm.weight":
                        LIB_LLAISYS.tensorLoad(self._w.mlp_norm_w[i], _ptr(t)); loaded += 1

                    # attention q/k/v/o
                    elif suffix == "self_attn.q_proj.weight":
                        LIB_LLAISYS.tensorLoad(self._w.attn_q_w[i], _ptr(t)); loaded += 1
                    elif suffix == "self_attn.q_proj.bias":
                        LIB_LLAISYS.tensorLoad(self._w.attn_q_b[i], _ptr(t)); loaded += 1

                    elif suffix == "self_attn.k_proj.weight":
                        LIB_LLAISYS.tensorLoad(self._w.attn_k_w[i], _ptr(t)); loaded += 1
                    elif suffix == "self_attn.k_proj.bias":
                        LIB_LLAISYS.tensorLoad(self._w.attn_k_b[i], _ptr(t)); loaded += 1

                    elif suffix == "self_attn.v_proj.weight":
                        LIB_LLAISYS.tensorLoad(self._w.attn_v_w[i], _ptr(t)); loaded += 1
                    elif suffix == "self_attn.v_proj.bias":
                        LIB_LLAISYS.tensorLoad(self._w.attn_v_b[i], _ptr(t)); loaded += 1

                    elif suffix == "self_attn.o_proj.weight":
                        LIB_LLAISYS.tensorLoad(self._w.attn_o_w[i], _ptr(t)); loaded += 1

                    # mlp
                    elif suffix == "mlp.gate_proj.weight":
                        LIB_LLAISYS.tensorLoad(self._w.mlp_gate_w[i], _ptr(t)); loaded += 1
                    elif suffix == "mlp.up_proj.weight":
                        LIB_LLAISYS.tensorLoad(self._w.mlp_up_w[i], _ptr(t)); loaded += 1
                    elif suffix == "mlp.down_proj.weight":
                        LIB_LLAISYS.tensorLoad(self._w.mlp_down_w[i], _ptr(t)); loaded += 1
                    else:
                        skipped += 1

    def __del__(self):
        try:
            if getattr(self, "_model", None):
                LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
                self._model = None
        except Exception:
            pass

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        tokens = list(inputs)
        if max_new_tokens is None:
            max_new_tokens = 1

        for _ in range(max_new_tokens):
            arr = (c_int64 * len(tokens))(*tokens)
            next_id = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, len(tokens))
            next_id = int(next_id)
            tokens.append(next_id)
            if next_id == int(self._meta.end_token):
                break

        return tokens



