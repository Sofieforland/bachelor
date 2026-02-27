# scripts/run_model.py
# kjør modell:
# cd /nfs/br1_student/dorteej/Bachelor/BachelorProject python3 -m bachelor.scripts.run_model --model qwen --n_rows 1 --write_csv
# sofie: python -m bachelor.scripts.run_model --model qwen --n_rows 4 --write_csv
from pathlib import Path
import argparse

from bachelor.models.qwen import QwenClient
from bachelor.models.llama import LlamaClient
from bachelor.models.medgemma import MedGemmaClient
from bachelor.pipeline.runner import run_file


#BASE_DIR = Path.home() / "Bachelor" / "BachelorProject" / "bachelor"
BASE_DIR = Path("/nfs/br1_student/sofiehf/bachelor")
IN_PATH = BASE_DIR / "outputs" / "dataset_with_notes.csv"

def main():
    parser = argparse.ArgumentParser(description="Run MDT panel on dataset")
    parser.add_argument("--model", choices=["qwen", "llama", "medgemma"], required=True)
    parser.add_argument("--n_rows", type=int, default=3, help="How many rows to run (default 3)")
    parser.add_argument("--write_csv", action="store_true", help="Also write CSV output")
    parser.add_argument("--append", action="store_true", help="Append to JSONL instead of overwriting")
    args = parser.parse_args()

    if args.model == "qwen":
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        model = QwenClient(model_id)
    elif args.model == "medgemma":
        model_id = "MedAIBase/MedGemma1.5:4b"
        model = MedGemmaClient(model_id)
    else:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = LlamaClient(model_id)

    out_jsonl = BASE_DIR / "outputs" / f"dataset_with_{args.model}_outputs.jsonl"
    out_csv = BASE_DIR / "outputs" / f"dataset_with_{args.model}_outputs.csv"

    run_file(
        model=model,
        in_path=IN_PATH,
        out_jsonl_path=out_jsonl,
        n_rows=args.n_rows,
        write_csv=args.write_csv,
        out_csv_path=out_csv if args.write_csv else None,
        append_jsonl=args.append,
        model_name=args.model,
        model_id=model_id,
    )

if __name__ == "__main__":
    main()