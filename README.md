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

# 4. (optional) Export an espeak-free sherpa-onnx bundle for on-device deployment
ghanavoice export-onnx --model ./my_voice/best.ckpt --language "Asante Twi" --out ./onnx
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

## Deploy (espeak-free sherpa-onnx)
```bash
ghanavoice export-onnx --model my_voice/best.ckpt --language "Asante Twi" --out onnx/
```
writes a self-contained, **espeak-free** bundle that sherpa-onnx runs end-to-end — no
`espeak`/`phonemizer` pip or apt package needed (sherpa-onnx has espeak-ng compiled in and
phonemizes with the bundled data):
```
onnx/
├── acoustic.onnx           # Matcha acoustic model (single-speaker; language baked in)
├── vocos-22khz-univ.onnx   # sherpa-onnx's pre-built universal Vocos vocoder (mel → STFT)
├── tokens.txt              # symbol → id (lfn/base symbol set)
├── espeak-ng-data/         # espeak-ng data incl. the lfn voice — so no system espeak is needed
├── metadata.json           # config + baked language
└── sherpa_infer.py         # runnable espeak-free demo (sherpa_onnx.OfflineTts)
```
Run it anywhere `sherpa-onnx` is installed (Python, or embed the same config in C++/Kotlin/Swift):
```bash
pip install sherpa-onnx soundfile
python onnx/sherpa_infer.py --text "Fa nifa so kɔ anim kakra." --out hello.wav
```
The language you pass to `export-onnx` is **baked into the acoustic model** (its speaker-slot
embedding becomes a constant), so the exported model is a plain single-speaker Matcha model —
exactly the shape sherpa-onnx's built-in Matcha frontend expects. sherpa-onnx phonemizes the text
itself (lfn voice, from `espeak-ng-data/`) and does the final ISTFT in C++. Export one bundle per
language.

## Status
- [x] 42-language registry, vendored Matcha engine
- [x] `ghanavoice prepare` — phonemize + mel + filter + stats (HF & local)
- [x] `ghanavoice train` — finetune loop + checkpointing + early stopping (+ optional HF push)
- [x] `ghanavoice synthesize` — inference; **pretrained Vocos by default** (HiFiGAN / finetuned Vocos optional)
- [x] `ghanavoice export-onnx` — export an **espeak-free** sherpa-onnx bundle for on-device deployment

## License / credits
Matcha-TTS code © its authors (see vendored headers). Base models by GhanaNLP Community.
