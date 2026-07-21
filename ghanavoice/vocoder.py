"""Vocoders for turning the model's mel-spectrogram into a waveform.

- hifigan: universal HiFiGAN (matches the mel config: 22.05kHz / 80-mel / fmax 8000),
  auto-downloaded. Good default, no finetuning needed.
- vocos: pretrained BSC-LT/vocos-mel-22khz, or a Ghana-finetuned {backbone,head} checkpoint.
"""
import torch

from matcha.utils.utils import get_user_data_dir, assert_model_downloaded

HIFIGAN_UNIV_URL = "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000"


def load_hifigan(device):
    from matcha.hifigan.config import v1
    from matcha.hifigan.env import AttrDict
    from matcha.hifigan.models import Generator as HiFiGAN
    from matcha.hifigan.denoiser import Denoiser

    path = get_user_data_dir() / "hifigan_univ_v1"
    assert_model_downloaded(path, HIFIGAN_UNIV_URL)
    net = HiFiGAN(AttrDict(v1)).to(device)
    net.load_state_dict(torch.load(path, map_location=device)["generator"])
    net.eval()
    net.remove_weight_norm()
    denoiser = Denoiser(net, mode="zeros")

    def vocode(mel):
        with torch.inference_mode():
            audio = net(mel).clamp(-1, 1)
            audio = denoiser(audio.squeeze(1), strength=2.5e-4).squeeze(1)
        return audio.squeeze().cpu().numpy()

    return vocode


def load_vocos(device, ckpt=None):
    """ckpt: None -> pretrained BSC-LT/vocos-mel-22khz; else a path/HF file to a Ghana-finetuned
    checkpoint saved as {"backbone":..., "head":..., "pretrained": <repo>}."""
    import yaml
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from vocos.pretrained import instantiate_class

    pretrained = "BSC-LT/vocos-mel-22khz"
    ft_state = None
    if ckpt is not None:
        ck_path = ckpt if Path(ckpt).exists() else hf_hub_download(*ckpt.split(":", 1)) if ":" in ckpt else ckpt
        ft = torch.load(ck_path, map_location=device, weights_only=False)
        pretrained = ft.get("pretrained", pretrained)
        ft_state = ft

    cfg = yaml.safe_load(open(hf_hub_download(pretrained, "config.yaml"), encoding="utf-8"))
    state = torch.load(hf_hub_download(pretrained, "pytorch_model.bin"), map_location=device, weights_only=False)
    backbone = instantiate_class((), cfg["backbone"]).to(device).eval()
    head = instantiate_class((), cfg["head"]).to(device).eval()
    if ft_state is not None:
        backbone.load_state_dict(ft_state["backbone"]); head.load_state_dict(ft_state["head"])
    else:
        backbone.load_state_dict({k[9:]: v for k, v in state.items() if k.startswith("backbone.")})
        head.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("head.")})

    def vocode(mel):
        with torch.inference_mode():
            return head(backbone(mel)).clamp(-1, 1).squeeze().cpu().numpy()

    return vocode


def get_vocoder(name, device, vocos_ckpt=None):
    if name == "hifigan":
        return load_hifigan(device)
    if name == "vocos":
        return load_vocos(device, vocos_ckpt)
    raise ValueError(f"Unknown vocoder '{name}' (use 'hifigan' or 'vocos')")
