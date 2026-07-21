"""Vocoders for turning the model's mel-spectrogram into a waveform.

- hifigan: universal HiFiGAN (matches the mel config: 22.05kHz / 80-mel / fmax 8000),
  auto-downloaded. Good default, no finetuning needed.
- vocos: pretrained BSC-LT/vocos-mel-22khz, or a Ghana-finetuned {backbone,head} checkpoint.
"""
import torch

from matcha.utils.utils import get_user_data_dir, assert_model_downloaded

HIFIGAN_UNIV_URL = "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000"

# Ghana-finetuned Vocos vocoder (best sounding for these voices). Pulled from HF on demand.
GHANA_VOCOS_REPO = "ghananlpcommunity/ghana-speech-vocos"
GHANA_VOCOS_FILE = "last.pt"


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


def load_vocos(device, ft_ckpt=None, ft_repo=None, ft_file=GHANA_VOCOS_FILE):
    """Build a Vocos vocoder.

    - ft_ckpt: local path to a Ghana-finetuned checkpoint {backbone, head, pretrained}
    - ft_repo: HF repo id to download the finetuned checkpoint (ft_file) from
    - neither: the plain pretrained BSC-LT/vocos-mel-22khz
    """
    import yaml
    from huggingface_hub import hf_hub_download
    from vocos.pretrained import instantiate_class

    pretrained = "BSC-LT/vocos-mel-22khz"
    ft_state = None
    if ft_ckpt is None and ft_repo is not None:
        ft_ckpt = hf_hub_download(ft_repo, ft_file, repo_type="model")
    if ft_ckpt is not None:
        ft_state = torch.load(ft_ckpt, map_location=device, weights_only=False)
        pretrained = ft_state.get("pretrained", pretrained)

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


def get_vocoder(name, device, vocos_ckpt=None, vocos_repo=GHANA_VOCOS_REPO, vocos_file=GHANA_VOCOS_FILE):
    """name: 'vocos-ghana' (finetuned, default), 'vocos' (pretrained), 'hifigan'."""
    if name == "hifigan":
        return load_hifigan(device)
    if name == "vocos":
        return load_vocos(device)  # plain pretrained
    if name == "vocos-ghana":
        # local checkpoint wins; otherwise download the finetuned one from HF
        return load_vocos(device, ft_ckpt=vocos_ckpt, ft_repo=None if vocos_ckpt else vocos_repo, ft_file=vocos_file)
    raise ValueError(f"Unknown vocoder '{name}' (use 'vocos-ghana', 'vocos', or 'hifigan')")
