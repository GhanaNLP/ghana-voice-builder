"""Unified `ghanavoice` command:

    ghanavoice prepare     ...   prepare data (local folder or HF dataset)
    ghanavoice train       ...   finetune the base model
    ghanavoice synthesize  ...   generate speech
    ghanavoice export-onnx ...   export to ONNX (sherpa-onnx / on-device)
    ghanavoice languages         list supported languages
"""
import sys

USAGE = __doc__


def _list_languages():
    from ghanavoice.languages import LANGUAGES
    print(f"{'id':>3}  {'iso':<5} {'language'}")
    print("-" * 34)
    for lid, (_reg, iso, human) in LANGUAGES.items():
        print(f"{lid:>3}  {iso:<5} {human}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        return
    cmd = sys.argv.pop(1)  # remove subcommand so each module's argparse sees clean args
    sys.argv[0] = f"ghanavoice {cmd}"
    if cmd == "prepare":
        from ghanavoice.prepare import main as m; m()
    elif cmd == "train":
        from ghanavoice.train import main as m; m()
    elif cmd in ("synthesize", "synth", "infer"):
        from ghanavoice.infer import main as m; m()
    elif cmd in ("export-onnx", "export_onnx", "onnx"):
        from ghanavoice.export_onnx import main as m; m()
    elif cmd == "languages":
        _list_languages()
    else:
        print(f"Unknown command '{cmd}'.\n{USAGE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
