# Ghana Voice Builder 🇬🇭🎙️

Finetune a strong multilingual **base TTS model** on a *small* amount of your own Ghanaian-language
speech and get a great-sounding voice — for **any of the 42 supported languages**, one or many at
a time. Clone, point it at your data (a HuggingFace dataset **or** a local folder with a
`metadata.csv`), run a couple of commands, and synthesize.

Built on [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) (flow-matching acoustic model),
with a shared phonemizer + per-language conditioning so a single base model covers all languages.
The model code is **vendored** in this repo — no external model dependency to install.

> Status: **under construction.** The base model + engine are in place; the `prepare` → `train`
> → `synthesize` CLI is being built and validated incrementally (see Roadmap).

---

## Why finetune instead of train from scratch?
The base model has already learned Ghanaian phonetics, prosody, and 42 language identities from
hundreds of hours of speech. Finetuning adapts it to *your* speaker/dialect/recording conditions
with as little as ~15–30 minutes of audio per language — far less data and compute than training
from zero.

## Supported languages
42 languages (Akuapem/Asante Twi, Fante, Ewe, Dagbani, Hausa, Nzema, Gonja, Kasem, Kusaal, …).
Run `ghanavoice languages` for the full list. Each has a fixed id used for conditioning.

---

## Quickstart (target UX)

```bash
git clone https://github.com/michsethowusu/ghana-voice-builder
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

Audio: mono WAV, any sample rate (resampled internally). ~15+ min per language recommended.

---

## Base model
Default: [`ghananlpcommunity/ghana-speech-nano-langtok`](https://huggingface.co/ghananlpcommunity/ghana-speech-nano-langtok)
(downloaded automatically on first `train`). Override with `--base-model`.

## Roadmap
- [x] Repo scaffold, 42-language registry, vendored Matcha engine
- [ ] `ghanavoice prepare` — phonemize + mel + filter + stats (HF & local)
- [ ] `ghanavoice train` — finetune loop + checkpointing + early stopping
- [ ] `ghanavoice synthesize` — inference (HiFiGAN vocoder; optional finetuned Vocos)
- [ ] Optional: vocoder finetuning for extra naturalness

## License / credits
Matcha-TTS code © its authors (see vendored headers). Base models by GhanaNLP Community.
