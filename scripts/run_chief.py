from pathlib import Path
import argparse
import torch
#python3 -m scripts.run_chief --model llama --n_rows 1

from models.qwen import QwenClient
from models.llama import LlamaClient
from models.medgemma import MedGemmaClient
from pipeline.runner_chief import run_chief_file


BASE_DIR = Path.home() / "Bachelor" / "BachelorProject" / "bachelor"


def main():
    parser = argparse.ArgumentParser(description="Run chief on merged panel outputs")
    parser.add_argument("--model", choices=["qwen", "llama", "medgemma"], required=True)
    parser.add_argument("--n_rows", type=int, default=None, help="How many rows to run (default: all)")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    IN_PATH = BASE_DIR / "outputs" / "Merged" / f"{args.model}_GPs.jsonl"

    if args.model == "qwen":
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        model = QwenClient(model_id, torch_dtype=torch.float16)
    elif args.model == "medgemma":
        model_id = "MedAIBase/MedGemma1.5:4b"
        model = MedGemmaClient(model_id)
    else:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = LlamaClient(model_id, torch_dtype=torch.float16)

    out_jsonl = BASE_DIR / "outputs" / f"chief_outputs_{args.model}.jsonl"

    # run_chief_file(
    #     model=model,
    #     in_path=IN_PATH,
    #     out_path=out_jsonl,
    #     append_jsonl=args.append,
    #     n_rows=args.n_rows,
    # )

    run_chief_file(
    model=model,
    in_path=Path("outputs/Merged/llama_GPs.jsonl"),
    out_path=Path("outputs/No_reputation_chiefs/chief_outputs_llama.jsonl"),
    append_jsonl=True,
)



if __name__ == "__main__":
    main()