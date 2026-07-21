"""Export a finetuned Ghana Voice model to ONNX for on-device / sherpa-onnx deployment.

Produces (in --out):
  acoustic.onnx          Matcha acoustic model: (x, x_lengths, scales[temp,length_scale], spks) -> mel
  vocos-22khz-univ.onnx  sherpa-onnx's pre-built universal Vocos vocoder: mel -> wav (22.05 kHz)
  tokens.txt      symbol<TAB>id   (how text characters map to token ids)
  metadata.json   mel config, sample rate, language ids, and the tokenization recipe

Note on sherpa-onnx: the two .onnx graphs run under ONNX Runtime (that's what sherpa-onnx's
vocoder runner uses). Our text frontend (lfn phonemizer + a language token + blank interspersing)
is non-standard, so text->token must be done with the recipe in metadata.json / `onnx_infer.py`
rather than sherpa-onnx's built-in Matcha frontend.
"""
import argparse
import json
from pathlib import Path

import torch

_orig_load = torch.load
def _load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)
torch.load = _load

from matcha.models.matcha_tts import MatchaTTS  # noqa: E402
from matcha.text.symbols import symbols, lang_token_id, N_BASE_SYMBOLS, N_LANGS  # noqa: E402

SR, NFFT, NMELS, HOP, WIN, FMIN, FMAX = 22050, 1024, 80, 256, 1024, 0, 8000


def export_acoustic(model, out_path, n_timesteps, opset):
    def onnx_forward(x, x_lengths, scales, spks=None):
        out = model.synthesise(x, x_lengths, n_timesteps, scales[0], spks, scales[1])
        return out["mel"], out["mel_lengths"]
    model.forward = onnx_forward
    x = torch.randint(1, 50, (1, 40), dtype=torch.long)
    x_lengths = torch.LongTensor([40])
    scales = torch.Tensor([0.3, 1.0])
    spks = torch.LongTensor([2])
    model.to_onnx(
        str(out_path), (x, x_lengths, scales, spks),
        input_names=["x", "x_lengths", "scales", "spks"],
        output_names=["mel", "mel_lengths"],
        dynamic_axes={"x": {0: "b", 1: "t"}, "x_lengths": {0: "b"},
                      "mel": {0: "b", 2: "t"}, "mel_lengths": {0: "b"}, "spks": {0: "b"}},
        opset_version=opset, export_params=True, do_constant_folding=True,
    )


# sherpa-onnx ships a pre-built, universal Vocos vocoder ONNX (mel->wav, 22.05 kHz). We reuse it
# rather than exporting our own (Vocos' ISTFT head doesn't export cleanly), which also guarantees
# sherpa-onnx compatibility.
SHERPA_VOCODER_REPO = "k2-fsa/sherpa-onnx-models"
SHERPA_VOCODER_REV = "6eebd0a85f1be93dd7e9bdb461efbdff6d193f04"
SHERPA_VOCODER_FILE = "vocoder-models/vocos-22khz-univ.onnx"


def fetch_sherpa_vocoder(out_dir):
    import shutil
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SHERPA_VOCODER_REPO, SHERPA_VOCODER_FILE, revision=SHERPA_VOCODER_REV, repo_type="model")
    dst = out_dir / "vocos-22khz-univ.onnx"
    shutil.copy(p, dst)
    return dst


def add_sherpa_metadata(acoustic_path):
    """Tag the acoustic ONNX with the metadata sherpa-onnx reads to recognize a Matcha model."""
    import onnx
    m = onnx.load(str(acoustic_path))
    tags = {
        "model_type": "matcha", "sample_rate": str(SR), "n_speakers": str(N_LANGS),
        "add_blank": "1", "voice": "lfn", "version": "1", "language": "multi",
        "comment": "ghana-voice-builder; frontend = lfn phonemes + language-token prepend (see metadata.json)",
    }
    for k, v in tags.items():
        e = m.metadata_props.add(); e.key = k; e.value = v
    onnx.save(m, str(acoustic_path))


def main():
    p = argparse.ArgumentParser(description="Export a Ghana Voice model to ONNX (sherpa-onnx ready).")
    p.add_argument("--model", required=True, help="Finetuned (or base) .ckpt")
    p.add_argument("--out", required=True, help="Output directory for the ONNX bundle")
    p.add_argument("--n-timesteps", type=int, default=10)
    p.add_argument("--opset", type=int, default=17, help="ONNX opset (>=17 for Vocos ISTFT)")
    p.add_argument("--no-vocoder", action="store_true", help="Export only the acoustic model")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = MatchaTTS.load_from_checkpoint(a.model, map_location="cpu").eval()

    print("[export] acoustic -> acoustic.onnx")
    export_acoustic(model, out / "acoustic.onnx", a.n_timesteps, a.opset)
    add_sherpa_metadata(out / "acoustic.onnx")

    if not a.no_vocoder:
        print("[export] fetching sherpa-onnx pre-built vocoder -> vocos-22khz-univ.onnx")
        fetch_sherpa_vocoder(out)

    # tokens.txt
    with open(out / "tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(symbols):
            f.write(f"{s}\t{i}\n")

    # metadata + tokenization recipe
    meta = {
        "sample_rate": SR, "n_mels": NMELS, "n_fft": NFFT, "hop": HOP, "win": WIN,
        "fmin": FMIN, "fmax": FMAX, "n_timesteps": a.n_timesteps,
        "n_base_symbols": N_BASE_SYMBOLS, "n_languages": N_LANGS, "n_vocab": len(symbols),
        "acoustic_inputs": ["x (token ids)", "x_lengths", "scales=[temperature,length_scale]", "spks=[lang_id]"],
        "tokenization": (
            "text -> twi_cleaners (lfn IPA) -> map chars to ids via tokens.txt -> intersperse blank(0) "
            "-> prepend language token id (N_BASE_SYMBOLS + lang_id). spks = [lang_id]."
        ),
        "lang_token_id_formula": f"{N_BASE_SYMBOLS} + lang_id",
    }
    meta["vocoder_note"] = (
        "vocos-22khz-univ.onnx outputs STFT (mag, x, y); reconstruct waveform via ISTFT "
        "(n_fft=1024, hop=256, win=1024, hann). sherpa-onnx does this ISTFT in its runtime; "
        "onnx_infer.py does it in Python."
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    (out / "onnx_infer.py").write_text(_ONNX_INFER_DEMO)
    print(f"[export] wrote tokens.txt + metadata.json + onnx_infer.py -> {out}")


_ONNX_INFER_DEMO = '''#!/usr/bin/env python3
"""Standalone ONNX inference for a Ghana Voice export bundle.

    python onnx_infer.py --language "Asante Twi" --text "Akwaaba!" --out hello.wav

Needs: onnxruntime, torch, soundfile, and the ghanavoice/matcha package (for tokenization).
"""
import argparse
from pathlib import Path
import numpy as np, onnxruntime as ort, soundfile as sf, torch
from matcha.text import text_to_sequence
from matcha.text.symbols import lang_token_id
from matcha.text.cleaners import twi_cleaners
from matcha.utils.utils import intersperse
from ghanavoice.languages import resolve

HERE = Path(__file__).parent
_ac = ort.InferenceSession(str(HERE / "acoustic.onnx"))
_vo = ort.InferenceSession(str(HERE / "vocos-22khz-univ.onnx"))


def synth(text, language, temperature=0.3, length_scale=1.0):
    lid = resolve(language)
    seq, _ = text_to_sequence(twi_cleaners(text), ["twi_phonemes"])
    tokens = [lang_token_id(lid)] + intersperse(seq, 0)
    mel, _ = _ac.run(["mel", "mel_lengths"], {
        "x": np.array([tokens], np.int64), "x_lengths": np.array([len(tokens)], np.int64),
        "scales": np.array([temperature, length_scale], np.float32), "spks": np.array([lid], np.int64)})
    mag, xr, yi = _vo.run(["mag", "x", "y"], {"mels": mel})
    S = torch.from_numpy(mag) * torch.complex(torch.from_numpy(xr), torch.from_numpy(yi))
    return torch.istft(S, 1024, 256, 1024, torch.hann_window(1024), center=True).squeeze().numpy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--language", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--temperature", type=float, default=0.3)
    a = p.parse_args()
    sf.write(a.out, synth(a.text, a.language, a.temperature), 22050)
    print("wrote", a.out)
'''


if __name__ == "__main__":
    main()
