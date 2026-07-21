# Ghana Voice Builder 🇬🇭🎙️

Finetune a strong multilingual **base TTS model** on a *small* amount of your own Ghanaian-language
speech and get a great-sounding voice — for **any of the 42 supported languages**, one or many at
a time. Clone, point it at your data (a HuggingFace dataset **or** a local folder with a
`metadata.csv`), run a couple of commands, and synthesize.

Built on [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) (flow-matching acoustic model),
with a shared phonemizer + per-language conditioning so a single base model covers all languages.
The model code is **vendored** in this repo — no external model dependency to install.

---

## Why finetune instead of train from scratch?
The base model has already learned Ghanaian phonetics, prosody, and 42 language identities from
hundreds of hours of speech. Finetuning adapts it to *your* speaker/dialect/recording conditions
with about ~5 hours of audio per language (recommended minimum) — far less data and compute than training
from zero.

## Supported languages
42 languages (Akuapem/Asante Twi, Fante, Ewe, Dagbani, Hausa, Nzema, Gonja, Kasem, Kusaal, …).
Run `ghanavoice languages` for the full list. Each has a fixed id used for conditioning.

---

## Quickstart (target UX)

```bash
git clone https://github.com/GhanaNLP/ghana-voice-builder
cd ghana-voice-builder
pip install -e .            # installs deps + builds the alignment extension

# 1. Prepare your data (local folder OR HuggingFace dataset)
ghanavoice prepare --input ./my_data --out ./prepped
#   or:  ghanavoice prepare --hf-dataset myorg/my-twi-speech --out ./prepped

# 2. Finetune the base model on it
ghanavoice train --data ./prepped --out ./my_voice

# 3. Synthesize
ghanavoice synthesize --model ./my_voice/best.ckpt \
    --language "Asante Twi" --text "Akwaaba!" --out hello.wav

# 4. (optional) Export to ONNX for on-device / sherpa-onnx deployment
ghanavoice export-onnx --model ./my_voice/best.ckpt --out ./onnx
```

Multiple languages: just include rows for each language in your data — the base model handles
them jointly, and you pick the language at synthesis time with `--language`.

---

## Data format

**Local folder:**
```
my_data/
├── metadata.csv
└── wavs/
    ├── clip0001.wav
    └── ...
```
`metadata.csv` (pipe-separated, no header):
```
wavs/clip0001.wav|Akwaaba, wo ho te sɛn?|Asante Twi
wavs/clip0002.wav|Ɛyɛ, meda wo ase.|Asante Twi
```
Columns: `audio_path | text | language`. `language` accepts an id, ISO code, or name
(e.g. `2`, `twi`, `Asante Twi`). Text is raw orthography — phonemization is automatic.

**HuggingFace dataset:** any dataset with `audio`, `text`, and a `language` column (column names
configurable). See `ghanavoice prepare --help`.

Audio: mono WAV, any sample rate (resampled internally). **~5 hours per language recommended** (less works — you just get a friendly heads-up).

---

## Base model & vocoder
- **Acoustic base model** (finetuned by `train`): [`ghananlpcommunity/ghana-speech-nano`](https://huggingface.co/ghananlpcommunity/ghana-speech-nano),
  downloaded automatically. Override with `--base-model`.
- **Vocoder** (mel → waveform at synthesis): `synthesize` uses the **pretrained Vocos**
  (`BSC-LT/vocos-mel-22khz`) by default — downloaded automatically. Use `--vocoder hifigan` for the
  universal HiFiGAN (no download auth needed), or `--vocoder vocos-ghana` for a Ghana-finetuned Vocos.

The base model is public and downloaded automatically — no setup needed.

## Deploy (ONNX / sherpa-onnx)
`ghanavoice export-onnx --model my_voice/best.ckpt --out onnx/` writes a self-contained bundle:
```
onnx/
├── acoustic.onnx           # Matcha acoustic model (tagged with sherpa-onnx metadata)
├── vocos-22khz-univ.onnx   # sherpa-onnx's pre-built universal Vocos vocoder (mel → STFT)
├── tokens.txt              # symbol → id
├── metadata.json           # mel config, language ids, tokenization recipe
└── onnx_infer.py           # runnable demo: text → audio via ONNX Runtime
```
Run it anywhere ONNX Runtime is available (Python demo, or embed in an app):
```bash
python onnx/onnx_infer.py --language "Asante Twi" --text "Akwaaba!" --out hello.wav
```
Note: the vocoder ONNX emits the STFT (`mag,x,y`); the final **ISTFT** is done by the runtime
(sherpa-onnx does it in C++; `onnx_infer.py` does it in Python). The text frontend (lfn phonemes
+ a language token) is ours, so tokenization follows `metadata.json`/`onnx_infer.py` rather than
sherpa-onnx's built-in Matcha frontend.

## Status
- [x] 42-language registry, vendored Matcha engine
- [x] `ghanavoice prepare` — phonemize + mel + filter + stats (HF & local)
- [x] `ghanavoice train` — finetune loop + checkpointing + early stopping (+ optional HF push)
- [x] `ghanavoice synthesize` — inference; **pretrained Vocos by default** (HiFiGAN / finetuned Vocos optional)
- [x] `ghanavoice export-onnx` — export to ONNX for on-device / sherpa-onnx deployment

## License / credits
Matcha-TTS code © its authors (see vendored headers). Base models by GhanaNLP Community.
