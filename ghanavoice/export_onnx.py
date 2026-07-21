"""Export a finetuned Ghana Voice model to a sherpa-onnx bundle for espeak-free, on-device TTS.

Produces (in --out) a bundle sherpa-onnx runs end-to-end with NO espeak pip/apt dependency
(sherpa-onnx has espeak-ng compiled in and phonemizes using the bundled espeak-ng-data/):

  acoustic.onnx           Matcha acoustic model (tagged with sherpa-onnx metadata: voice=lfn, ...)
  vocos-22khz-univ.onnx   sherpa-onnx's pre-built universal Vocos vocoder
  tokens.txt              symbol -> id (lfn/base symbol set)
  espeak-ng-data/         espeak-ng data incl. the lfn voice (so no system espeak is needed)
  sherpa_infer.py         espeak-free demo using sherpa_onnx.OfflineTts
  metadata.json           config + language-id table

At synthesis, pick the language via the speaker slot: sherpa `tts.generate(text, sid=<lang_id>)`.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

_orig_load = torch.load
def _load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)
torch.load = _load

from matcha.models.matcha_tts import MatchaTTS  # noqa: E402
from ghanavoice.languages import resolve, name as lang_name  # noqa: E402

SR, NMELS = 22050, 80
SHERPA_VOCODER = ("k2-fsa/sherpa-onnx-models", "6eebd0a85f1be93dd7e9bdb461efbdff6d193f04",
                  "vocoder-models/vocos-22khz-univ.onnx")
FRONTEND_REPO = "ghananlpcommunity/nano-twi"  # reuse its lfn tokens.txt + espeak-ng-data


def export_acoustic(model, out_path, n_timesteps, lang_id, opset=17):
    # sherpa-onnx's Matcha runner is single-speaker (no `spks` input). A finetune targets one
    # language, so we bake that language's speaker-slot embedding in as a constant -> the ONNX
    # is a plain single-speaker Matcha model, exactly like the reference bundle.
    spk_const = torch.tensor([lang_id], dtype=torch.long)

    def onnx_forward(x, x_lengths, scales):
        out = model.synthesise(x, x_lengths, n_timesteps, scales[0], spk_const, scales[1])
        return out["mel"], out["mel_lengths"]
    model.forward = onnx_forward
    x = torch.randint(1, 50, (1, 40), dtype=torch.long)
    model.to_onnx(
        str(out_path), (x, torch.LongTensor([40]), torch.Tensor([0.3, 1.0])),
        input_names=["x", "x_lengths", "scales"], output_names=["mel", "mel_lengths"],
        dynamic_axes={"x": {0: "b", 1: "t"}, "x_lengths": {0: "b"},
                      "mel": {0: "b", 2: "t"}, "mel_lengths": {0: "b"}},
        opset_version=opset, export_params=True, do_constant_folding=True,
    )


def add_sherpa_metadata(acoustic_path, n_speakers, n_timesteps, language="multi"):
    import onnx
    m = onnx.load(str(acoustic_path))
    tags = {
        "model_type": "matcha-tts", "language": language, "voice": "lfn",
        "has_espeak": "1", "jieba": "0", "n_speakers": str(n_speakers),
        "sample_rate": str(SR), "version": "1", "pad_id": "0",
        "use_icefall": "0", "use_eos_bos": "0", "num_ode_steps": str(n_timesteps),
        "model_author": "GhanaNLP", "comment": "Ghana Voice Builder",
    }
    for k, v in tags.items():
        e = m.metadata_props.add(); e.key = k; e.value = v
    onnx.save(m, str(acoustic_path))


def fetch(out_dir):
    """Download the pre-built Vocos vocoder + the lfn frontend (tokens.txt + espeak-ng-data)."""
    from huggingface_hub import hf_hub_download, list_repo_files
    # vocoder
    repo, rev, f = SHERPA_VOCODER
    shutil.copy(hf_hub_download(repo, f, revision=rev, repo_type="model"), out_dir / "vocos-22khz-univ.onnx")
    # frontend: tokens.txt + espeak-ng-data (identical lfn setup)
    shutil.copy(hf_hub_download(FRONTEND_REPO, "sherpa-onnx/tokens.txt", repo_type="model"), out_dir / "tokens.txt")
    for rf in list_repo_files(FRONTEND_REPO, repo_type="model"):
        if rf.startswith("sherpa-onnx/espeak-ng-data/"):
            dst = out_dir / rf[len("sherpa-onnx/"):]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hf_hub_download(FRONTEND_REPO, rf, repo_type="model"), dst)


def main():
    p = argparse.ArgumentParser(description="Export a Ghana Voice model to a sherpa-onnx (espeak-free) bundle.")
    p.add_argument("--model", required=True, help="Finetuned (or base) .ckpt")
    p.add_argument("--out", required=True, help="Output bundle directory")
    p.add_argument("--n-timesteps", type=int, default=10)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--language", required=True,
                   help="The language this model was finetuned on (id / iso / name). "
                        "Its speaker-slot embedding is baked into the ONNX.")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lid = resolve(a.language)
    model = MatchaTTS.load_from_checkpoint(a.model, map_location="cpu").eval()

    print(f"[export] acoustic -> acoustic.onnx (language baked in: {lang_name(lid)}, slot {lid})")
    export_acoustic(model, out / "acoustic.onnx", a.n_timesteps, lid, a.opset)
    add_sherpa_metadata(out / "acoustic.onnx", 1, a.n_timesteps, lang_name(lid))

    print("[export] fetching vocoder + lfn frontend (tokens.txt, espeak-ng-data)")
    fetch(out)

    (out / "metadata.json").write_text(json.dumps({
        "sample_rate": SR, "n_speakers": 1, "n_timesteps": a.n_timesteps,
        "language": lang_name(lid), "baked_language_id": lid,
        "deploy": "sherpa-onnx (espeak-free); single-speaker, language baked in at export",
    }, indent=2, ensure_ascii=False))
    (out / "sherpa_infer.py").write_text(_SHERPA_DEMO)
    print(f"[export] sherpa-onnx bundle ready -> {out}")
    print("[export] try:  python sherpa_infer.py --text 'Akwaaba' --sid 2 --out hi.wav")


_SHERPA_DEMO = '''#!/usr/bin/env python3
"""Espeak-free TTS via sherpa-onnx (no phonemizer/espeak pip or apt needed).

    pip install sherpa-onnx soundfile
    python sherpa_infer.py --text "Akwaaba, wo ho te sɛn?" --out hello.wav

The language was baked in at export time, so this is a single-speaker model (sid=0).
"""
import argparse
from pathlib import Path
import sherpa_onnx, soundfile as sf

HERE = Path(__file__).parent


def build():
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(HERE / "acoustic.onnx"),
                vocoder=str(HERE / "vocos-22khz-univ.onnx"),
                lexicon="", tokens=str(HERE / "tokens.txt"),
                data_dir=str(HERE / "espeak-ng-data")),
            num_threads=2)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--sid", type=int, default=0, help="single-speaker model; leave at 0")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    audio = build().generate(a.text, sid=a.sid, speed=a.speed)
    sf.write(a.out, audio.samples, audio.sample_rate)
    print("wrote", a.out)
'''


if __name__ == "__main__":
    main()
