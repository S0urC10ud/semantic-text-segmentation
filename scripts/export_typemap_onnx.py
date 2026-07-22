"""Build dynamic-length ONNX graphs for the typemap U-Net and Mamba models.

Build-time tool only (needs ``onnx``; NOT a runtime dependency of the package).
Reads the slimmed weights already bundled at ``packages/typemap/typemap/data``
and writes ``unet_al.onnx`` / ``mamba_al.onnx`` next to them.

The graphs consume *compact* token ids (0..129); the runtime applies the compact
remap before feeding the session. Math mirrors ``typemap._numpy_backend`` exactly.

  * U-Net : input ``ids[N, 1536]`` int64  -> ``logits[N, 1536, 35]``
            (fixed 1536 window, dynamic batch; runtime tiles arbitrary length)
  * Mamba : input ``ids[T]`` int64         -> ``logits[T, 35]``
            (dynamic sequence length via the ONNX ``Scan`` op)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DATA = Path(__file__).resolve().parents[1] / "packages" / "typemap" / "typemap" / "data"
OPSET = 17
GELU_C = float(np.sqrt(2.0 / np.pi))


def _load(npz_name: str) -> dict:
    d = np.load(DATA / npz_name)
    return {k.replace("__", "/"): np.asarray(d[k], dtype=np.float32) for k in d.files}


class GB:
    """Tiny ONNX graph builder: unique names, initializers, nodes."""

    def __init__(self):
        self.nodes = []
        self.inits = []
        self._n = 0
        self._seen = set()

    def name(self, stem: str) -> str:
        self._n += 1
        return f"{stem}_{self._n}"

    def init(self, arr: np.ndarray, name: str) -> str:
        if name not in self._seen:
            self.inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
            self._seen.add(name)
        return name

    def const(self, arr: np.ndarray, stem: str) -> str:
        return self.init(arr, self.name(stem))

    def node(self, op: str, ins, outs=None, stem=None, **attrs):
        if outs is None:
            outs = [self.name(stem or op.lower())]
        self.nodes.append(helper.make_node(op, ins, outs, **attrs))
        return outs[0] if len(outs) == 1 else outs


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


# --------------------------------------------------------------------------
# shared ops
# --------------------------------------------------------------------------
def gelu(g: GB, x: str) -> str:
    half = g.const(_f32(0.5), "c_half")
    coef = g.const(_f32(0.044715), "c_coef")
    cc = g.const(_f32(GELU_C), "c_gelu")
    one = g.const(_f32(1.0), "c_one")
    x2 = g.node("Mul", [x, x], stem="x2")
    x3 = g.node("Mul", [x2, x], stem="x3")
    cx3 = g.node("Mul", [x3, coef], stem="cx3")
    inner = g.node("Add", [x, cx3], stem="inner")
    cinner = g.node("Mul", [inner, cc], stem="cinner")
    t = g.node("Tanh", [cinner], stem="tanh")
    onet = g.node("Add", [t, one], stem="onet")
    hx = g.node("Mul", [x, half], stem="hx")
    return g.node("Mul", [hx, onet], stem="gelu")


def silu(g: GB, x: str) -> str:
    s = g.node("Sigmoid", [x], stem="sig")
    return g.node("Mul", [x, s], stem="silu")


def groupnorm(g: GB, x: str, scale: np.ndarray, bias: np.ndarray, C: int, L: int,
              groups: int = 8, eps: float = 1e-5) -> str:
    """x: [N, C, L] -> normalised [N, C, L]; reduce over (in-group ch, L) per group."""
    cpg = C // groups
    shp = g.const(np.array([0, groups, cpg, L], dtype=np.int64), "gn_shape")
    xr = g.node("Reshape", [x, shp], stem="gn_r")  # [N, g, cpg, L]
    mean = g.node("ReduceMean", [xr], stem="gn_mean", axes=[2, 3], keepdims=1)
    xc = g.node("Sub", [xr, mean], stem="gn_xc")
    sq = g.node("Mul", [xc, xc], stem="gn_sq")
    var = g.node("ReduceMean", [sq], stem="gn_var", axes=[2, 3], keepdims=1)
    epsc = g.const(_f32(eps), "gn_eps")
    vare = g.node("Add", [var, epsc], stem="gn_vare")
    std = g.node("Sqrt", [vare], stem="gn_std")
    norm = g.node("Div", [xc, std], stem="gn_norm")
    back = g.const(np.array([0, C, L], dtype=np.int64), "gn_back")
    nr = g.node("Reshape", [norm, back], stem="gn_nr")  # [N, C, L]
    sc = g.const(scale.reshape(1, C, 1), "gn_scale")
    bi = g.const(bias.reshape(1, C, 1), "gn_bias")
    scaled = g.node("Mul", [nr, sc], stem="gn_scaled")
    return g.node("Add", [scaled, bi], stem="gn_out")


def layernorm(g: GB, x: str, scale: np.ndarray, bias: np.ndarray, eps: float = 1e-6) -> str:
    sc = g.const(scale, "ln_scale")
    bi = g.const(bias, "ln_bias")
    return g.node("LayerNormalization", [x, sc, bi], stem="ln", axis=-1, epsilon=eps)


def dense(g: GB, x: str, kernel: np.ndarray, bias: np.ndarray, stem="dense") -> str:
    k = g.const(kernel, stem + "_k")
    b = g.const(bias, stem + "_b")
    mm = g.node("MatMul", [x, k], stem=stem + "_mm")
    return g.node("Add", [mm, b], stem=stem + "_add")


# --------------------------------------------------------------------------
# U-Net
# --------------------------------------------------------------------------
def build_unet(w: dict, channels, num_classes: int, seq: int = 1536) -> onnx.ModelProto:
    g = GB()
    cb = [0]

    def convblock(x: str, cin: int, cout: int, L: int) -> str:
        p = f"ConvBlock1D_{cb[0]}/"
        cb[0] += 1
        kern = np.transpose(w[p + "Conv_0/kernel"], (2, 1, 0))  # (k,cin,cout)->(cout,cin,k)
        W = g.const(kern, "cv_w")
        B = g.const(w[p + "Conv_0/bias"], "cv_b")
        conv = g.node("Conv", [x, W, B], stem="conv",
                      kernel_shape=[3], pads=[1, 1], strides=[1], group=1)
        gn = groupnorm(g, conv, w[p + "GroupNorm_0/scale"], w[p + "GroupNorm_0/bias"], cout, L)
        return gelu(g, gn)

    ids = "ids"
    emb_w = g.const(w["Embed_0/embedding"], "embed")
    emb = g.node("Gather", [emb_w, ids], stem="emb", axis=0)        # [N,L,256]
    x = g.node("Transpose", [emb], stem="x0", perm=[0, 2, 1])        # [N,256,L]

    lengths = [seq >> i for i in range(len(channels))]               # 1536..12
    in_ch = w["Embed_0/embedding"].shape[1]
    skips = []
    cur = in_ch
    for i, ch in enumerate(channels):
        L = lengths[i]
        x = convblock(x, cur, ch, L); cur = ch
        x = convblock(x, cur, ch, L)
        skips.append((x, ch, L))
        if i < len(channels) - 1:
            x = g.node("MaxPool", [x], stem="pool", kernel_shape=[2], strides=[2])

    for i, ch in enumerate(reversed(channels[:-1])):
        scales = g.const(_f32([1.0, 1.0, 2.0]), "rs_scale")
        roi = g.const(_f32([]), "rs_roi")
        x = g.node("Resize", [x, roi, scales], stem="up",
                   mode="nearest", coordinate_transformation_mode="asymmetric",
                   nearest_mode="floor")
        skip_x, skip_ch, L = skips[-(i + 2)]
        x = g.node("Concat", [x, skip_x], stem="cat", axis=1)        # [N, cur+skip_ch, L]
        cur = cur + skip_ch
        x = convblock(x, cur, ch, L); cur = ch
        x = convblock(x, cur, ch, L)

    hk = np.transpose(w["Conv_0/kernel"], (2, 1, 0))                 # (1,cin,cls)->(cls,cin,1)
    HW = g.const(hk, "head_w")
    HB = g.const(w["Conv_0/bias"], "head_b")
    head = g.node("Conv", [x, HW, HB], stem="head", kernel_shape=[1], pads=[0, 0], strides=[1], group=1)
    logits = g.node("Transpose", [head], outs=["logits"], perm=[0, 2, 1])  # [N,L,cls]

    inp = helper.make_tensor_value_info("ids", TensorProto.INT64, ["N", seq])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", seq, num_classes])
    graph = helper.make_graph(g.nodes, "typemap_unet", [inp], [out], g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=9)
    model.doc_string = "typemap U-Net (fixed 1536 window, dynamic batch)"
    onnx.checker.check_model(model)
    return model


# --------------------------------------------------------------------------
# Mamba
# --------------------------------------------------------------------------
def _scan_body(idx: int, reverse: bool, di: int, ds: int) -> onnx.GraphProto:
    """Recurrence body: in (s_prev, dt_t, u_t, B_t, C_t) -> out (s_next, y_t).

    References outer initializers ``A_neg_{idx}`` [di,ds] and ``D_{idx}`` [di].
    """
    tag = f"b{idx}_{'r' if reverse else 'f'}"
    n = []
    ax0 = numpy_helper.from_array(np.array([0], dtype=np.int64), f"{tag}_ax0")
    ax1 = numpy_helper.from_array(np.array([1], dtype=np.int64), f"{tag}_ax1")

    def nd(op, ins, out, **attrs):
        n.append(helper.make_node(op, ins, [out], **attrs))
        return out

    A = f"A_neg_{idx}"
    D = f"D_{idx}"
    nd("Unsqueeze", ["dt_t", f"{tag}_ax1"], f"{tag}_dtc")          # [di,1]
    nd("Mul", [f"{tag}_dtc", A], f"{tag}_ae")                      # [di,ds]
    nd("Exp", [f"{tag}_ae"], f"{tag}_a")                           # a_t
    nd("Unsqueeze", ["u_t", f"{tag}_ax1"], f"{tag}_uc")            # [di,1]
    nd("Unsqueeze", ["B_t", f"{tag}_ax0"], f"{tag}_br")            # [1,ds]
    nd("Mul", [f"{tag}_dtc", f"{tag}_br"], f"{tag}_dtb")           # [di,ds]
    nd("Mul", [f"{tag}_uc", f"{tag}_dtb"], f"{tag}_b")             # b_t
    nd("Mul", [f"{tag}_a", "s_prev"], f"{tag}_as")
    nd("Add", [f"{tag}_as", f"{tag}_b"], "s_next")                 # state out [di,ds]
    nd("Unsqueeze", ["C_t", f"{tag}_ax0"], f"{tag}_cr")            # [1,ds]
    nd("Mul", ["s_next", f"{tag}_cr"], f"{tag}_sc")
    nd("ReduceSum", [f"{tag}_sc", f"{tag}_ax1"], f"{tag}_ys", keepdims=0)  # [di]
    nd("Mul", ["u_t", D], f"{tag}_ud")                            # [di]
    nd("Add", [f"{tag}_ys", f"{tag}_ud"], "y_t")                  # scan out [di]

    s_prev = helper.make_tensor_value_info("s_prev", TensorProto.FLOAT, [di, ds])
    dt_t = helper.make_tensor_value_info("dt_t", TensorProto.FLOAT, [di])
    u_t = helper.make_tensor_value_info("u_t", TensorProto.FLOAT, [di])
    B_t = helper.make_tensor_value_info("B_t", TensorProto.FLOAT, [ds])
    C_t = helper.make_tensor_value_info("C_t", TensorProto.FLOAT, [ds])
    s_next = helper.make_tensor_value_info("s_next", TensorProto.FLOAT, [di, ds])
    y_t = helper.make_tensor_value_info("y_t", TensorProto.FLOAT, [di])
    return helper.make_graph(n, f"scan_{tag}",
                             [s_prev, dt_t, u_t, B_t, C_t], [s_next, y_t],
                             [ax0, ax1])


def _scan(g: GB, idx: int, reverse: bool, dt: str, u: str, B: str, C: str,
          di: int, ds: int) -> str:
    body = _scan_body(idx, reverse, di, ds)
    s0 = g.const(np.zeros((di, ds), dtype=np.float32), f"s0_{idx}_{int(reverse)}")
    d = [1, 1, 1, 1] if reverse else [0, 0, 0, 0]
    outs = [g.name("s_final"), g.name("y_scan")]
    g.nodes.append(helper.make_node(
        "Scan", [s0, dt, u, B, C], outs, body=body, num_scan_inputs=4,
        scan_input_axes=[0, 0, 0, 0], scan_input_directions=d,
        scan_output_axes=[0], scan_output_directions=[1 if reverse else 0]))
    return outs[1]  # y [T, di]


def build_mamba(w: dict, n_layers: int, d_state: int, dt_rank: int,
                d_conv: int, num_classes: int) -> onnx.ModelProto:
    g = GB()
    ids = "ids"
    emb_w = g.const(w["Embed_0/embedding"], "embed")
    h = g.node("Gather", [emb_w, ids], stem="emb", axis=0)          # [T, dm]
    dm = w["Embed_0/embedding"].shape[1]

    for i in range(n_layers):
        p = f"CheckpointMambaBlock1D_{i}/"
        di = w[p + "Dense_0/kernel"].shape[1] // 2
        # bake A_neg and D as named initializers (referenced by scan bodies)
        g.init(_f32(-np.exp(w[p + "A_log"])), f"A_neg_{i}")
        g.init(_f32(w[p + "D"]), f"D_{i}")

        ln = layernorm(g, h, w[p + "LayerNorm_0/scale"], w[p + "LayerNorm_0/bias"], eps=1e-6)
        xz = dense(g, ln, w[p + "Dense_0/kernel"], w[p + "Dense_0/bias"], stem=f"d0_{i}")
        split2 = g.const(np.array([di, di], dtype=np.int64), f"sp2_{i}")
        u, gate = g.node("Split", [xz, split2], outs=[g.name("u"), g.name("gate")], axis=1)

        # depthwise conv1d (SAME, k=d_conv): [T,di]->[di,T]->[1,di,T]->Conv->[T,di]
        ut = g.node("Transpose", [u], stem="ut", perm=[1, 0])
        ub = g.node("Unsqueeze", [ut, g.const(np.array([0], dtype=np.int64), "cv_ax0")], stem="ub")
        ck = np.transpose(w[p + "Conv_0/kernel"], (2, 1, 0))        # (k,1,di)->(di,1,k)
        CW = g.const(ck, f"mc_w_{i}")
        CB = g.const(w[p + "Conv_0/bias"], f"mc_b_{i}")
        low = (d_conv - 1) // 2
        uc = g.node("Conv", [ub, CW, CB], stem="mconv",
                    kernel_shape=[d_conv], pads=[low, d_conv - 1 - low], strides=[1], group=di)
        ucs = g.node("Squeeze", [uc, g.const(np.array([0], dtype=np.int64), "cv_ax0b")], stem="ucs")
        uconv = g.node("Transpose", [ucs], stem="uconv", perm=[1, 0])
        usilu = silu(g, uconv)

        xdbl = dense(g, usilu, w[p + "Dense_1/kernel"], w[p + "Dense_1/bias"], stem=f"d1_{i}")
        split3 = g.const(np.array([dt_rank, d_state, d_state], dtype=np.int64), f"sp3_{i}")
        dt_raw, Bm, Cm = g.node("Split", [xdbl, split3],
                                outs=[g.name("dtr"), g.name("Bm"), g.name("Cm")], axis=1)
        dt = dense(g, dt_raw, w[p + "Dense_2/kernel"], w[p + "Dense_2/bias"], stem=f"d2_{i}")
        sp = g.node("Softplus", [dt], stem="sp")
        dt_sp = g.node("Add", [sp, g.const(_f32(1e-4), "dt_eps")], stem="dtsp")

        yf = _scan(g, i, False, dt_sp, usilu, Bm, Cm, di, d_state)
        yr = _scan(g, i, True, dt_sp, usilu, Bm, Cm, di, d_state)
        y = g.node("Add", [yf, yr], stem="ybi")
        gs = silu(g, gate)
        yg = g.node("Mul", [y, gs], stem="yg")
        yout = dense(g, yg, w[p + "Dense_3/kernel"], w[p + "Dense_3/bias"], stem=f"d3_{i}")
        h = g.node("Add", [h, yout], stem=f"res_{i}")

    h = layernorm(g, h, w["LayerNorm_0/scale"], w["LayerNorm_0/bias"], eps=1e-6)
    k = g.const(w["Dense_0/kernel"], "head_k")
    b = g.const(w["Dense_0/bias"], "head_b")
    mm = g.node("MatMul", [h, k], stem="head_mm")
    g.node("Add", [mm, b], outs=["logits"])

    inp = helper.make_tensor_value_info("ids", TensorProto.INT64, ["T"])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["T", num_classes])
    graph = helper.make_graph(g.nodes, "typemap_mamba", [inp], [out], g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=9)
    model.doc_string = "typemap Mamba (dynamic sequence length via Scan)"
    onnx.checker.check_model(model)
    return model


def main():
    man = json.loads((DATA / "manifest.json").read_text())
    num_classes = man["num_classes"]

    uw = _load(man["unet"]["file"])
    unet = build_unet(uw, man["unet"]["channels"], num_classes, seq=man["unet"]["window_bytes"])
    onnx.save(unet, DATA / "unet_al.onnx")
    print("wrote unet_al.onnx")

    mw = _load(man["mamba"]["file"])
    mc = man["mamba"]
    mamba = build_mamba(mw, mc["n_layers"], mc["d_state"], mc["dt_rank"], mc["d_conv"], num_classes)
    onnx.save(mamba, DATA / "mamba_al.onnx")
    print("wrote mamba_al.onnx")


if __name__ == "__main__":
    main()
