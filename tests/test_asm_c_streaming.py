import pytest
import torch

from aletheion_state_models.variants import build_compact_streaming
from drm_language_emitter import DRMConfig, DRMEmitterModel
from scripts.run_mqar_architecture_comparison import load_asm_c_with_wider_memory


def config() -> DRMConfig:
    return DRMConfig(vocab_size=17,d_token=8,d_state=12,n_directions=4,metric_rank=2,hidden_size=16,sequence_mode="directional_block_cumsum",directional_cumsum_step_mode="velocity",directional_cumsum_block_size=4,directional_local_mixer="causal_conv",directional_local_mixer_hidden_size=12,directional_local_mixer_kernel_size=3,token_state_residual=True,selective_memory=True,selective_memory_hidden_size=12,bounded_state=False,use_direction_field=False)


def test_compact_streaming_matches_full_forward_fp32_and_drops_prefix():
    torch.manual_seed(7); model=build_compact_streaming(config()).eval(); tokens=torch.randint(0,17,(2,15)); expected=model(tokens,collect_diagnostics=False)["logits"]
    first,state=model.prefill(tokens[:,:3]); observed=[first]
    assert state.compact and state.input_ids.shape==(2,0) and state.sequence_length==3
    for position in range(3,tokens.shape[1]):
        logits,state=model.decode_step(tokens[:,position],state); observed.append(logits[:,None]); assert state.input_ids.numel()==0; assert state.block_tokens.shape[1]<state.block_size
    assert state.sequence_length==tokens.shape[1]
    assert torch.allclose(torch.cat(observed,1),expected,atol=1e-6,rtol=1e-6)


def test_compact_emitter_always_receives_one_position():
    model=build_compact_streaming(config()).eval(); seen=[]
    hook=model.emitter.register_forward_pre_hook(lambda _module,args: seen.append(tuple(args[0].shape)))
    _,state=model.prefill(torch.randint(0,17,(1,3))); seen.clear()
    for _ in range(12): _,state=model.decode_step(torch.randint(0,17,(1,)),state)
    hook.remove(); assert seen and {shape[1] for shape in seen}=={1}


def test_compact_cache_is_bounded_at_completed_blocks():
    model=build_compact_streaming(config()).eval(); _,state=model.prefill(torch.randint(0,17,(1,4))); sizes=[]
    for position in range(32):
        _,state=model.decode_step(torch.randint(0,17,(1,)),state)
        if state.sequence_length%state.block_size==0: sizes.append(sum(t.numel()*t.element_size() for t in (state.input_ids,state.completed_state,state.block_tokens)))
    assert len(set(sizes))==1


def test_compact_streaming_rejects_non_block_fallback():
    cfg=config(); cfg.sequence_mode="directional_cumsum"; cfg.compact_streaming_inference=True
    model=DRMEmitterModel(cfg.validated_copy()).eval()
    with pytest.raises(RuntimeError,match="fixed block boundaries"): model.prefill(torch.randint(0,17,(1,3)))


def test_wider_mqar_memory_reuses_every_non_memory_weight(tmp_path):
    base = build_compact_streaming(config())
    checkpoint = tmp_path / "asm_c.pt"
    torch.save({"config": base.config.to_dict(), "model": base.state_dict()}, checkpoint)
    widened, reset = load_asm_c_with_wider_memory(checkpoint)
    assert widened.config.selective_memory_hidden_size == 24
    assert reset and all(key.startswith("selective_memory.") for key in reset)
    for key, value in base.state_dict().items():
        if not key.startswith("selective_memory."):
            assert torch.equal(widened.state_dict()[key], value)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA BF16 required")
def test_compact_bf16_argmax_parity_is_measured():
    model=build_compact_streaming(config()).cuda().eval(); tokens=torch.randint(0,17,(1,12),device="cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16): expected=model(tokens,collect_diagnostics=False)["logits"]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        first,state=model.prefill(tokens[:,:3]); rows=[first]
        for position in range(3,tokens.shape[1]): logits,state=model.decode_step(tokens[:,position],state); rows.append(logits[:,None])
    actual=torch.cat(rows,1); assert float((actual.float()-expected.float()).abs().max())<0.05; assert torch.equal(actual.argmax(-1),expected.argmax(-1))
