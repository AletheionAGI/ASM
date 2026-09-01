"""Resume PMCS-64 after sealed language training with memory-heavy specialization."""
import json
from pathlib import Path
import torch
from aletheion_state_models.benchmarks.phase3a_checkpoint import write_result
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits
from aletheion_state_models.benchmarks.phase3a_training import Phase3ARunResult
from aletheion_state_models.benchmarks.purpose_mqar import specialize_memory_heavy
from aletheion_state_models.benchmarks.purpose_plots import render_purpose_charts
from aletheion_state_models.benchmarks.purpose_streaming import probe_streaming
from aletheion_state_models.benchmarks.purpose_summary import summarize_purpose_suite
from aletheion_state_models.benchmarks.purpose_variants import PURPOSE_VARIANTS, build_purpose_variant, parameter_inventory

ROOT=Path("runs/asm_cm_vs_vr_s_pmcs64");OUTPUT=Path("docs/benchmarks/asm_cm_vs_vr_s_pmcs64");SEEDS=(17,29,43)


def load_model(variant,seed,device):
    checkpoint=torch.load(ROOT/variant/f"seed_{seed}"/"best.pt",map_location=device,weights_only=False);model,_=build_purpose_variant(variant,seed);model.load_state_dict(checkpoint["model_state"]);return model.to(device).eval()


def main():
    device="cuda";splits=load_document_hash_splits(Path("data/wikipedia_en_20231101_sample.txt"));language=[];mqar=[];streaming=[]
    for seed in SEEDS:
        for variant in PURPOSE_VARIANTS:
            directory=ROOT/variant/f"seed_{seed}";language.append(Phase3ARunResult(**json.loads((directory/"result.json").read_text())))
            result_path=directory/"mqar_result.json"
            if result_path.exists():
                previous=json.loads(result_path.read_text())
                if previous.get("protocol") != "80pct_mqar_20pct_language":
                    write_result(directory/"mqar_light_result.json",previous)
            if result_path.exists() and json.loads(result_path.read_text()).get("protocol") == "80pct_mqar_20pct_language": result=json.loads(result_path.read_text())
            else:
                print(f"memory-heavy {variant} seed={seed}",flush=True);result=specialize_memory_heavy(load_model(variant,seed,device),splits,variant=variant,seed=seed,output_directory=ROOT,device=device)
            mqar.append(result)
    for seed in SEEDS:
        for variant in PURPOSE_VARIANTS:
            path=ROOT/variant/f"seed_{seed}"/"streaming_result.json"
            if path.exists(): result=json.loads(path.read_text())
            else:
                lengths=(512,4096,32768) if seed==17 else (512,4096);print(f"streaming {variant} seed={seed}",flush=True);rows=probe_streaming(load_model(variant,seed,device),lengths,seed=50_000+seed,device=device);result={"variant":variant,"seed":seed,"streaming":[dict(row,seed=seed) for row in rows]};write_result(path,result)
            streaming.append(result)
    manifest=json.loads((OUTPUT/"manifest.json").read_text());manifest["mqar_protocol"]={"mix":"80% MQAR / 20% language replay","curriculum":{"40":1000,"80":500,"160":500,"320":300,"512":200,"1024":100,"4096":25},"selection":"fixed terminal checkpoint; no test-based selection"};write_result(OUTPUT/"manifest.json",manifest)
    summary=summarize_purpose_suite(language,mqar,streaming,parameter_inventory());write_result(OUTPUT/"summary.json",summary);render_purpose_charts(summary,OUTPUT);print(json.dumps({"gates":summary["gates"],"language_winner":summary["language_winner"],"mqar_winner":summary["mqar_32k_winner"]},indent=2));raise SystemExit(0 if summary["technical_passed"] else 1)


if __name__=="__main__":main()
