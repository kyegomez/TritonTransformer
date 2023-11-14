import pytest
import torch
import triton
import triton.language as tl

from tt.attention import attention
from tt.feedforward import (
    TritonFeedForward,
)

# Check if CUDA is available for testing
cuda_available = torch.cuda.is_available()


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.fixture
def setup_feedforward():
    # Sample dimensions for testing
    input_features = 64
    hidden_features = 128
    output_features = 64
    batch_size = 10
    return (
        TritonFeedForward(input_features, hidden_features, output_features),
        batch_size,
        input_features,
        output_features,
    )


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_output_shape(setup_feedforward):
    model, batch_size, input_features, output_features = setup_feedforward
    model.to("cuda")
    input_tensor = torch.randn(batch_size, input_features, device="cuda")
    output = model(input_tensor)
    assert output.shape == (batch_size, output_features), "Output shape is incorrect"


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_different_batch_sizes(setup_feedforward):
    model, _, input_features, output_features = setup_feedforward
    model.to("cuda")
    for batch_size in [1, 5, 20]:
        input_tensor = torch.randn(batch_size, input_features, device="cuda")
        output = model(input_tensor)
        assert output.shape == (
            batch_size,
            output_features,
        ), f"Output shape is incorrect for batch size {batch_size}"


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_gradient_flow(setup_feedforward):
    model, batch_size, input_features, _ = setup_feedforward
    model.to("cuda")
    input_tensor = torch.randn(
        batch_size, input_features, device="cuda", requires_grad=True
    )
    output = model(input_tensor)
    assert output.requires_grad, "Gradients are not flowing through the model"
    (output.sum()).backward()  # Trigger backward pass
    assert input_tensor.grad is not None, "Gradients not computed for input tensor"


# Attention tests


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale).half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0)


try:
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func as flash_attn_func,
    )

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 15)],
                line_arg="provider",
                line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
                line_names=["Triton"] + (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-")],
                ylabel="ms",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "D_HEAD": D_HEAD,
                    "dtype": torch.float16,
                    "mode": mode,
                    "causal": causal,
                },
            )
        )


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"
):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    if provider == "triton":
        q = torch.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        k = torch.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        if mode == "fwd" and TORCH_HAS_FP8:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
        v = torch.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn(
            (BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True
        )
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


# only works on post-Ampere GPUs right now
bench_flash_attention.run(save_path=".", print_data=True)
