from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.pipeline import build_and_save_index, load_index
from engine.query_engine import answer_query


def cmd_build_index(args: argparse.Namespace) -> None:
    Path(Path(args.index_path).parent).mkdir(parents=True, exist_ok=True)
    build_and_save_index(data_dir=args.data_dir, index_path=args.index_path)
    print(f"Index saved to: {args.index_path}")


def cmd_ask(args: argparse.Namespace) -> None:
    idx = load_index(args.index_path)
    out = answer_query(
        idx,
        args.query,
        top_k=args.top_k,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        llm_model_path=args.llm_model_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(out["answer"])
    if args.show_context:
        print("\n=== RETRIEVED CONTEXT ===")
        for hit in out["retrieved"]:
            print(f'\n--- {hit["source"]} (score={hit["score"]:.4f}) ---\n{hit["text"]}\n')


def cmd_eval(args: argparse.Namespace) -> None:
    tmp_index = args.index_path or "artifacts/index_eval.pkl"
    Path("artifacts").mkdir(exist_ok=True, parents=True)
    build_and_save_index(data_dir=args.data_dir, index_path=tmp_index)
    idx = load_index(tmp_index)

    with open(args.questions, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = []
    for q in payload.get("questions", []):
        results.append(
            answer_query(
                idx,
                q,
                top_k=args.top_k,
                use_llm=args.use_llm,
                llm_provider=args.llm_provider,
                llm_model_path=args.llm_model_path,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        )

    out = {"results": results}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote: {args.out}")
    else:
        print(json.dumps(out, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index")
    b.add_argument("--data-dir", required=True)
    b.add_argument("--index-path", required=True)
    b.set_defaults(func=cmd_build_index)

    a = sub.add_parser("ask")
    a.add_argument("--index-path", required=True)
    a.add_argument("--query", required=True)
    a.add_argument("--top-k", type=int, default=8)
    a.add_argument("--show-context", action="store_true")
    a.add_argument("--json", action="store_true")

    a.add_argument("--use-llm", action="store_true")
    a.add_argument("--llm-provider", default="hf", choices=["hf"])
    a.add_argument("--llm-model-path", default="models/tinyllama", help="Local folder path containing model files")
    a.add_argument("--temperature", type=float, default=0.2)
    a.add_argument("--max-tokens", type=int, default=700)
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval")
    e.add_argument("--data-dir", required=True)
    e.add_argument("--questions", required=True)
    e.add_argument("--top-k", type=int, default=8)
    e.add_argument("--out", default="")
    e.add_argument("--index-path", default="")

    e.add_argument("--use-llm", action="store_true")
    e.add_argument("--llm-provider", default="hf", choices=["hf"])
    e.add_argument("--llm-model-path", default="models/tinyllama", help="Local folder path containing model files")
    e.add_argument("--temperature", type=float, default=0.2)
    e.add_argument("--max-tokens", type=int, default=700)
    e.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
