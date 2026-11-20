"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import json

from app import core


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="quantized-legal-tagger-monitor",
        description=(
            "Legal aid workers use this to evaluate the size and inference speed of a "
            "quantized legal clause tagger, ensuring efficient document processing on local systems."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("prepare", help="Milestone 1: prepare the LEDGAR dataset subset")
    sub.add_parser("train", help="Milestone 2: fine-tune GPT-2 for legal clause tagging")
    sub.add_parser("evaluate", help="Milestone 3: evaluate baseline FP32 model accuracy and size")
    sub.add_parser("benchmark", help="Milestones 4-5: quantize and compare model variants")
    sub.add_parser("dashboard", help="Milestone 6: display the model efficiency dashboard")

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Quantized Legal Tagger Monitor is ready. Open README.md for your milestones.")
    elif args.command == "prepare":
        print(json.dumps(core.milestone_1(), indent=2))
    elif args.command == "train":
        print(json.dumps(core.milestone_2(), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(core.milestone_3(), indent=2))
    elif args.command == "benchmark":
        print(json.dumps(core.milestone_5(), indent=2))
    elif args.command == "dashboard":
        core.milestone_6()


if __name__ == "__main__":
    main()