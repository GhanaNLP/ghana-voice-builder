"""Synthesize speech from a finetuned (or base) Ghana Voice model.

    ghanavoice synthesize --model my_voice/best.ckpt --language "Asante Twi" \
        --text "Akwaaba!" --out hello.wav
"""
import argparse
from pathlib import Path

import torch
import soundfile as sf

_orig_load = torch.load
def _load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)
torch.load = _load

from matcha.models.matcha_tts import MatchaTTS  # noqa: E402
from matcha.text import text_to_sequence  # noqa: E402
from matcha.text.symbols import lang_token_id  # noqa: E402
from matcha.text.cleaners import twi_cleaners  # noqa: E402
from matcha.utils.utils import intersperse  # noqa: E402
from ghanavoice.languages import resolve, name as lang_name  # noqa: E402
from ghanavoice.vocoder import get_vocoder  # noqa: E402

SR = 22050


def main():
    p = argparse.ArgumentParser(description="Synthesize speech with a Ghana Voice model.")
    p.add_argument("--model", required=True, help="Finetuned (or base) .ckpt")
    p.add_argument("--language", required=True, help="Language id / iso code / name")
    p.add_argument("--text", required=True, help="Text to speak (raw orthography)")
    p.add_argument("--out", required=True, help="Output .wav path")
    p.add_argument("--vocoder", default="hifigan", choices=["hifigan", "vocos"])
    p.add_argument("--vocos-ckpt", default=None, help="Finetuned Vocos checkpoint (path or HF file)")
    p.add_argument("--n-timesteps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--length-scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lid = resolve(a.language)

    model = MatchaTTS.load_from_checkpoint(a.model, map_location=device).to(device).eval()
    vocode = get_vocoder(a.vocoder, device, a.vocos_ckpt)

    seq, _ = text_to_sequence(twi_cleaners(a.text), ["twi_phonemes"])
    tokens = [lang_token_id(lid)] + intersperse(seq, 0)
    x = torch.tensor(tokens, dtype=torch.long, device=device)[None]
    x_len = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    spks = torch.tensor([lid], dtype=torch.long, device=device)

    with torch.inference_mode():
        out = model.synthesise(x, x_len, n_timesteps=a.n_timesteps, temperature=a.temperature,
                               spks=spks, length_scale=a.length_scale)
    audio = vocode(out["mel"])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.out, audio, SR)
    print(f"[synth] {lang_name(lid)}: wrote {a.out} ({len(audio) / SR:.1f}s)")


if __name__ == "__main__":
    main()
