"""Data preparation: turn raw speech+text (local folder or HuggingFace dataset) into the
phonemized, mel-cached, filtered form the finetuner consumes.

Output layout (--out):
    mels/<clip_id>.npy      raw 22.05kHz log-mel (80 x T), matching the base model
    train.txt               clip_id|lang_id|phonemes
    val.txt                 (held-out, a few clips per language)
    stats.json              mel_mean, mel_std, counts, languages
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from matcha.utils.audio import mel_spectrogram
from matcha.text.cleaners import twi_cleaners
from ghanavoice.languages import resolve, name as lang_name

SR, NFFT, NMELS, HOP, WIN, FMIN, FMAX = 22050, 1024, 80, 256, 1024, 0, 8000


def compute_mel(wave_22k):
    mel = mel_spectrogram(wave_22k.unsqueeze(0), NFFT, NMELS, SR, HOP, WIN, FMIN, FMAX, center=False)
    return mel.squeeze(0).numpy().astype(np.float32)  # (80, T)


def load_wave(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.T)  # (channels, samples)
    if wave.shape[0] > 1:
        wave = wave.mean(0, keepdim=True)
    wave = wave.squeeze(0)
    if sr != SR:
        wave = torchaudio.functional.resample(wave, sr, SR)
    return wave


def iter_local(input_dir):
    """Yield (clip_id, wave_or_path, text, language) from a folder + metadata.csv."""
    input_dir = Path(input_dir)
    meta = input_dir / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"No metadata.csv in {input_dir}")
    with open(meta, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) < 3 or not row[0].strip():
                continue
            apath = (input_dir / row[0].strip()).resolve()
            yield Path(apath).stem, apath, row[1].strip(), row[2].strip()


def iter_hf(dataset_id, split, audio_col, text_col, lang_col):
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    # If lang_col isn't a real column, treat it as a single fixed language for every row
    # (common for single-language datasets: --language-column "Asante Twi").
    has_lang = lang_col in ds.column_names
    for i, row in enumerate(ds):
        a = row[audio_col]
        wave = torch.from_numpy(np.asarray(a["array"], dtype=np.float32))
        sr = a["sampling_rate"]
        if sr != SR:
            wave = torchaudio.functional.resample(wave, sr, SR)
        cid = Path(a.get("path") or f"{dataset_id.split('/')[-1]}_{i:06d}").stem
        language = str(row[lang_col]) if has_lang else lang_col
        yield cid, ("__wave__", wave), row[text_col], language


def main():
    p = argparse.ArgumentParser(description="Prepare Ghanaian speech data for finetuning.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Local folder containing metadata.csv + audio")
    src.add_argument("--hf-dataset", help="HuggingFace dataset id (e.g. myorg/twi-speech)")
    p.add_argument("--out", required=True, help="Output directory for prepped data")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--text-column", default="text")
    p.add_argument("--language-column", default="language",
                   help="Column with the language, OR a fixed language for the whole dataset")
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=15.0)
    p.add_argument("--val-per-lang", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()

    out = Path(a.out)
    (out / "mels").mkdir(parents=True, exist_ok=True)

    if a.input:
        source = iter_local(a.input)
    else:
        source = iter_hf(a.hf_dataset, a.hf_split, a.audio_column, a.text_column, a.language_column)

    by_lang = defaultdict(list)
    secs_by_lang = defaultdict(float)
    mel_sum = mel_sqsum = mel_count = 0.0
    kept = skipped = 0
    langs_seen = set()

    RECOMMENDED_HOURS = 5.0

    for cid, audio_ref, text, language in tqdm(source, desc="prepare"):
        # language may be a per-row value or (for local) a fixed column; resolve flexibly
        try:
            lid = resolve(language)
        except KeyError:
            # allow the local --language-column to be a fixed language name for all rows
            lid = resolve(a.language_column)
        langs_seen.add(lid)

        try:
            if isinstance(audio_ref, tuple) and audio_ref[0] == "__wave__":
                wave = audio_ref[1]
            else:
                wave = load_wave(audio_ref)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[prepare] skip {cid}: cannot read audio ({e})")
            skipped += 1
            continue

        dur = wave.shape[-1] / SR
        if dur < a.min_duration or dur > a.max_duration:
            skipped += 1
            continue

        phon = twi_cleaners(text)
        if not phon.strip():
            skipped += 1
            continue

        mel = compute_mel(wave)
        np.save(out / "mels" / f"{cid}.npy", mel)
        mel_sum += float(mel.sum()); mel_sqsum += float((mel ** 2).sum()); mel_count += mel.size
        by_lang[lid].append((cid, lid, phon))
        secs_by_lang[lid] += dur
        kept += 1

    if kept == 0:
        raise SystemExit("[prepare] No clips kept — check --input/--hf-dataset, columns, and durations.")

    mel_mean = mel_sum / mel_count
    mel_std = float(np.sqrt(mel_sqsum / mel_count - mel_mean ** 2))

    rng = random.Random(a.seed)
    train_rows, val_rows = [], []
    for lid, rows in by_lang.items():
        rng.shuffle(rows)
        nv = min(a.val_per_lang, max(0, len(rows) - 1))
        val_rows += rows[:nv]
        train_rows += rows[nv:]

    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for cid, lid, phon in rows:
                f.write(f"{cid}|{lid}|{phon}\n")

    write(out / "train.txt", train_rows)
    write(out / "val.txt", val_rows)
    stats = {
        "mel_mean": float(mel_mean), "mel_std": float(mel_std),
        "n_train": len(train_rows), "n_val": len(val_rows),
        "languages": sorted({lang_name(l) for l in langs_seen}),
        "language_ids": sorted(langs_seen),
        "sample_rate": SR, "n_mels": NMELS, "fmax": FMAX,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    total_hours = sum(secs_by_lang.values()) / 3600
    print(f"\n[prepare] kept={kept} skipped={skipped}")
    print(f"[prepare] train={len(train_rows)} val={len(val_rows)} across {len(langs_seen)} language(s), "
          f"{total_hours:.2f}h total")
    print(f"[prepare] mel_mean={mel_mean:.4f} mel_std={mel_std:.4f}")
    print(f"[prepare] -> {out}")

    # Friendly heads-up (not a block): we recommend ~5h per language for a good voice.
    low = {lid: secs / 3600 for lid, secs in secs_by_lang.items() if secs / 3600 < RECOMMENDED_HOURS}
    if low:
        print(f"\n[prepare] 👋 heads-up: the recommended minimum is ~{RECOMMENDED_HOURS:.0f} hours "
              f"per language for the best-sounding voice. These are below that:")
        for lid in sorted(low, key=int):
            print(f"           • {lang_name(lid):<18} {low[lid]:.2f}h")
        print("           You can still go ahead and train — it'll work, quality may just be lower. "
              "More data helps.")


if __name__ == "__main__":
    main()
