"""Finetune the multilingual base model on prepped data.

Warm-starts from the base checkpoint (its full 220-token / 42-language architecture is
unchanged, so no surgery is needed), normalizes with the base mel stats, and trains a plain
loop on MatchaTTS.forward()'s losses (duration + prior + flow-matching). Early-stops on the
validation flow-matching (diff) loss — the metric that tracks acoustic quality.
"""
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Lightning checkpoints hold non-tensor objects; PyTorch>=2.6 defaults weights_only=True.
_orig_load = torch.load
def _load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)
torch.load = _load

from matcha.models.matcha_tts import MatchaTTS  # noqa: E402
from ghanavoice.data import FinetuneDataset, Collate  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def resolve_base_ckpt(base_model):
    """Local path -> use directly. Otherwise treat as an HF repo id and download a checkpoint."""
    if Path(base_model).exists():
        return base_model
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(base_model, repo_type="model")
    # Prefer the language-token variant; fall back to older layouts.
    for cand in ("langtok/last.ckpt", "checkpoints/last.ckpt", "last.ckpt"):
        if cand in files:
            return hf_hub_download(base_model, cand, repo_type="model")
    ckpts = [f for f in files if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found in {base_model}: {files}")
    return hf_hub_download(base_model, sorted(ckpts)[-1], repo_type="model")


def save_ckpt(model, path, epoch, val):
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": "2.0.0", "epoch": epoch, "val_diff_loss": val}, path)


def main():
    p = argparse.ArgumentParser(description="Finetune the Ghana Voice base model on prepped data.")
    p.add_argument("--data", required=True, help="Prepped directory (from `ghanavoice prepare`)")
    p.add_argument("--out", required=True, help="Output directory for checkpoints")
    p.add_argument("--base-model", default="ghananlpcommunity/ghana-speech-nano",
                   help="HF repo id or local .ckpt to warm-start from")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--smoke", type=int, default=0, help="run only N steps + val, then exit")
    a = p.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    data = Path(a.data)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    base_ckpt = resolve_base_ckpt(a.base_model)
    print(f"[train] warm-start from {base_ckpt}", flush=True)
    model = MatchaTTS.load_from_checkpoint(base_ckpt, map_location=device).to(device)
    mel_mean, mel_std = float(model.mel_mean), float(model.mel_std)
    out_size = model.out_size
    print(f"[train] base mel stats {mel_mean:.3f}/{mel_std:.3f}, out_size={out_size}", flush=True)

    tr = DataLoader(FinetuneDataset(data / "train.txt", data / "mels", mel_mean, mel_std),
                    batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                    collate_fn=Collate(), drop_last=False, persistent_workers=a.num_workers > 0,
                    pin_memory=True)
    val_rows = (data / "val.txt").read_text().strip()
    va = None
    if val_rows:
        va = DataLoader(FinetuneDataset(data / "val.txt", data / "mels", mel_mean, mel_std),
                        batch_size=a.batch_size, shuffle=False, num_workers=2, collate_fn=Collate())

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def run_forward(batch):
        b = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            dur, prior, diff, _ = model(b["x"], b["x_lengths"], b["y"], b["y_lengths"],
                                        spks=b["spks"], out_size=out_size, durations=None)
        return dur, prior, diff

    @torch.inference_mode()
    def validate():
        if va is None:
            return float("nan")
        model.eval(); tot = n = 0
        for batch in va:
            _, _, diff = run_forward(batch); tot += float(diff); n += 1
        model.train()
        return tot / max(n, 1)

    repo = os.environ.get("HF_REPO_ID")
    api = None
    if os.environ.get("HF_TOKEN") and repo:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        print(f"[train] pushing checkpoints to {repo}", flush=True)

    model.train()
    best, wait, step = float("inf"), 0, 0
    for epoch in range(a.max_epochs):
        for batch in tr:
            dur, prior, diff = run_forward(batch)
            loss = dur + prior + diff
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
            if step % a.log_every == 0:
                print(f"[train] e{epoch} step {step}: loss={float(loss):.3f} "
                      f"(dur={float(dur):.2f} prior={float(prior):.3f} diff={float(diff):.3f})", flush=True)
            if a.smoke and step >= a.smoke:
                print(f"[train] SMOKE ok: {step} steps, val diff={validate():.4f}", flush=True)
                save_ckpt(model, out / "smoke.ckpt", epoch, 0.0)
                print("[train] saved smoke.ckpt", flush=True)
                return

        v = validate()
        improved = v < best - 1e-4
        print(f"[train] epoch {epoch}: val diff={v:.4f} (best {best:.4f}) {'IMPROVED' if improved else ''}", flush=True)
        save_ckpt(model, out / "last.ckpt", epoch, v)
        if improved:
            best, wait = v, 0
            save_ckpt(model, out / "best.ckpt", epoch, v)
            if api:
                for f in ("best.ckpt", "last.ckpt"):
                    try:
                        api.upload_file(path_or_fileobj=str(out / f), path_in_repo=f, repo_id=repo,
                                        repo_type="model", commit_message=f"epoch {epoch} val {v:.4f}")
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"[train] HF upload failed ({e})", flush=True)
        else:
            wait += 1
            if wait >= a.patience:
                print(f"[train] early stop at epoch {epoch} (best val diff={best:.4f})", flush=True)
                break
    print(f"[train] done. best val diff={best:.4f}. checkpoints in {out}", flush=True)


if __name__ == "__main__":
    main()
