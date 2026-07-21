"""Trimmed utility helpers needed by the vendored Matcha model + inference.

The upstream matcha.utils.utils also contained Hydra/omegaconf/rich training helpers; those
are intentionally dropped here since Ghana Voice Builder drives training with a plain loop.
"""
import os
import sys
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def intersperse(lst, item):
    # Adds blank symbol between tokens
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def save_figure_to_numpy(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    return data


def plot_tensor(tensor):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return np.array(tensor)
    raise TypeError("Unsupported type for conversion to numpy array")


def get_user_data_dir(appname="matcha_tts"):
    home = os.environ.get("MATCHA_HOME")
    if home is not None:
        ans = Path(home).expanduser().resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    final_path = ans.joinpath(appname)
    final_path.mkdir(parents=True, exist_ok=True)
    return final_path


def assert_model_downloaded(checkpoint_path, url, use_wget=True):
    if Path(checkpoint_path).exists():
        print(f"[+] Model already present at {checkpoint_path}!")
        return
    print(f"[-] Model not found at {checkpoint_path}! Downloading ...")
    checkpoint_path = str(checkpoint_path)
    if use_wget:
        import wget
        wget.download(url=url, out=checkpoint_path)
    else:
        import gdown
        gdown.download(url=url, output=checkpoint_path, quiet=False, fuzzy=True)


def get_phoneme_durations(durations, phones):
    prev = durations[0]
    merged_durations = []
    for i in range(1, len(durations), 2):
        next_half = durations[i + 1] if i == len(durations) - 2 else ceil(durations[i + 1] / 2)
        curr = prev + durations[i] + next_half
        prev = durations[i + 1] - next_half
        merged_durations.append(curr)
    assert len(phones) == len(merged_durations)
    merged_durations = torch.cumsum(torch.tensor(merged_durations), 0, dtype=torch.long)
    start = torch.tensor(0)
    duration_json = []
    for i, duration in enumerate(merged_durations):
        duration_json.append(
            {phones[i]: {"starttime": start.item(), "endtime": duration.item(),
                         "duration": duration.item() - start.item()}}
        )
        start = duration
    return duration_json
