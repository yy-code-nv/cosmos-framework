# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""
Tests for UniAE S3 tokenizer (4x16x16).

Usage:
    # Basic encode/decode test with random data
    CUDA_VISIBLE_DEVICES=0 RUN_SKIPPED_TEST_LOCALLY=1 pytest -s cosmos_framework/model/vfm/tokenizers/uniae/noncausal_4x16x16_test.py -k test_uniae_s3

    # Full reconstruction test with real video (saves uniae_s3_recon.mp4)
    CUDA_VISIBLE_DEVICES=0 RUN_SKIPPED_TEST_LOCALLY=1 pytest -s cosmos_framework/model/vfm/tokenizers/uniae/noncausal_4x16x16_test.py -k test_local_video

Note: On this machine, CUDA device 0 = RTX 6000 Ada (48GB), device 1 = T400 (2GB).
      Always use CUDA_VISIBLE_DEVICES=0 for the RTX 6000.
"""

import inspect
import os

import numpy as np
import pytest
import torch

from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.utils.helper_test import RunIf
from cosmos_framework.model.tokenizer.models.sparse_autoencoder import AutoencoderKL
from cosmos_framework.configs.base.defaults.cluster import DefaultClusterConfig as CLUSTER_CONFIG
from cosmos_framework.configs.base.defaults.unittest import TOKENIZER_RECONSTRUCTION_VIDEO_PATH, UNITTEST_CONFIG
from cosmos_framework.model.vfm.tokenizers.uniae.noncausal_4x16x16 import _S1_ARCH, UniAEVAE
from cosmos_framework.model.vfm.tokenizers.unittest_utils import (
    numpy2tensor,
    pad_video_batch,
    tensor2numpy,
    unpad_video_batch,
)

UNIAE_S3_PATH = (
    "s3://bucket1/uniae/tok_experiments/"
    "uniae_s3_prod32_ditval_video_b1_50k_r1/checkpoints/iter_000050000.pt"
)


@pytest.mark.L0
def test_uniae_s1_arch_matches_autoencoder_signature() -> None:
    """The VFM wrapper should not pass stale tokenizer-training kwargs."""
    legacy_attention_keys = {
        "encoder_attn_mode",
        "encoder_window_size",
        "decoder_attn_mode",
        "decoder_window_size",
    }
    signature_keys = set(inspect.signature(AutoencoderKL.__init__).parameters)

    assert legacy_attention_keys.isdisjoint(_S1_ARCH)
    assert set(_S1_ARCH).issubset(signature_keys)
    assert _S1_ARCH["use_text_alignment"] is False
    assert _S1_ARCH["use_post_text_alignment"] is False


@pytest.mark.L0
@pytest.mark.parametrize(
    ("num_pixel_frames", "expected_latent_frames"),
    [
        (1, 1),
        (2, 1),
        (4, 1),
        (5, 2),
        (7, 2),
        (8, 2),
        (13, 4),
        (16, 4),
    ],
)
def test_uniae_latent_num_frames_matches_noncausal_padding(
    num_pixel_frames: int,
    expected_latent_frames: int,
) -> None:
    """Frame-count helper should match encode's pad-to-multiple behavior."""
    vae = UniAEVAE.__new__(UniAEVAE)
    vae._temporal_compression_factor = 4

    assert vae.get_latent_num_frames(num_pixel_frames) == expected_latent_frames


@pytest.mark.L0
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_uniae_s3():
    """Basic shape check: encode/decode with random data for a few T values."""
    vae = UniAEVAE(
        vae_pth=UNIAE_S3_PATH,
        object_store_credential_path_pretrained=CLUSTER_CONFIG.object_store_credential_pretrained,
        device="cuda",
        dtype=torch.bfloat16,
    )
    print(f"\n[UniAE S3] Model parameters: {vae.count_param() / 1e6:.2f}M")

    H, W = 256, 256
    for T in [4, 52, 100, 148]:
        video = torch.randn(1, 3, T, H, W, device="cuda", dtype=torch.bfloat16)
        latents = vae.encode(video)
        video_recon = vae.decode(latents)

        expected_T_latent = vae.get_latent_num_frames(T)
        assert latents.shape == (1, vae.z_dim, expected_T_latent, H // 16, W // 16), (
            f"T={T}: unexpected latent shape {tuple(latents.shape)}"
        )
        assert video_recon.shape == (1, 3, T, H, W), f"T={T}: unexpected recon shape {tuple(video_recon.shape)}"
        print(f"  T={T:3d}  latent={tuple(latents.shape[1:])}  recon={tuple(video_recon.shape[2:])}  OK")


@pytest.mark.L0
@RunIf(
    requires_file=[
        CLUSTER_CONFIG.object_store_credential_pretrained,
        UNITTEST_CONFIG.object_store_credential_data,
    ]
)
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_local_video():
    """Reconstruction test with real video for T in [52,56,...,148]; plots frame-wise PSNR per T."""
    import matplotlib.pyplot as plt

    vae = UniAEVAE(
        vae_pth=UNIAE_S3_PATH,
        object_store_credential_path_pretrained=CLUSTER_CONFIG.object_store_credential_pretrained,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Load enough frames to cover all T values
    T_values = list(range(52, 149, 4))  # 52, 56, ..., 148
    max_T = max(T_values)
    video_full = easy_io.load(
        os.path.join(f"s3://{UNITTEST_CONFIG.object_store_bucket_data}", TOKENIZER_RECONSTRUCTION_VIDEO_PATH),
        backend_args={
            "backend": "s3",
            "s3_credential_path": UNITTEST_CONFIG.object_store_credential_data,
        },
    )[0]  # [T_total, H, W, C]
    available_T = video_full.shape[0]
    print(f"\n[UniAE S3] Video loaded: {video_full.shape}, using T up to {min(max_T, available_T)}")

    os.makedirs("logs", exist_ok=True)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(14, 6))
    mean_psnrs = []

    for i, T in enumerate(T_values):
        if T > available_T:
            print(f"  T={T}: skipped (video only has {available_T} frames)")
            continue

        video_in_numpy = video_full[:T]  # [T, H, W, C]

        padded_video_batch, crop_region = pad_video_batch(
            video_in_numpy[np.newaxis, ...],
            temporal_align=4,
            spatial_align=16,
            causal_mode=False,
            only_pad_end=True,
        )
        video_tensor = numpy2tensor(padded_video_batch).cuda()

        latents = vae.encode(video_tensor)
        video_recon = vae.decode(latents)

        video_recon_numpy = tensor2numpy(video_recon)
        video_recon_unpadded = unpad_video_batch(video_recon_numpy, crop_region)

        gt = video_in_numpy[: video_recon_unpadded.shape[1]].astype(np.float32)
        recon = video_recon_unpadded[0].astype(np.float32)
        mse_per_frame = np.mean((gt - recon) ** 2, axis=(1, 2, 3))
        psnr_per_frame = 10 * np.log10(255**2 / np.maximum(mse_per_frame, 1e-10))
        mean_psnr = float(psnr_per_frame.mean())
        mean_psnrs.append((T, mean_psnr))

        assert mean_psnr >= 30.0, f"T={T}: mean PSNR {mean_psnr:.2f} dB < 30 dB threshold"
        color = cmap(i / len(T_values))
        ax.plot(psnr_per_frame, color=color, alpha=0.7, linewidth=0.8, label=f"T={T} ({mean_psnr:.1f}dB)")
        print(f"  T={T:3d}  latent={tuple(latents.shape[1:])}  mean PSNR={mean_psnr:.2f} dB")

        # Save reconstructed video
        video_path = f"logs/uniae_s3_recon_T{T:03d}.mp4"
        easy_io.dump(video_recon_unpadded[0].astype("uint8"), video_path)
        print(f"    saved {video_path}")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("UniAE S3 frame-wise PSNR — real video, T ∈ [52, 148] step 4")
    ax.legend(loc="upper right", fontsize=6, ncol=4)
    fig.tight_layout()
    plot_path = "logs/uniae_s3_local_framewise_psnr.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[UniAE S3] Plot saved to {plot_path}")
    if mean_psnrs:
        psnrs = [p for _, p in mean_psnrs]
        print(f"[UniAE S3] Mean PSNR range: {min(psnrs):.2f} – {max(psnrs):.2f} dB")


"""
Usage:
    CUDA_VISIBLE_DEVICES=0 RUN_SKIPPED_TEST_LOCALLY=1 pytest -s cosmos_framework/model/vfm/tokenizers/uniae/noncausal_4x16x16_test.py -k test_local_image
"""


@pytest.mark.L0
@RunIf(
    requires_file=[
        CLUSTER_CONFIG.object_store_credential_pretrained,
        UNITTEST_CONFIG.object_store_credential_data,
    ]
)
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_local_image():
    """Image reconstruction test — repeats image to 4 frames for non-causal tokenizer."""
    from PIL import Image

    vae = UniAEVAE(
        vae_pth=UNIAE_S3_PATH,
        object_store_credential_path_pretrained=CLUSTER_CONFIG.object_store_credential_pretrained,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Load first frame of test video as image
    video_in_numpy = easy_io.load(
        os.path.join(f"s3://{UNITTEST_CONFIG.object_store_bucket_data}", TOKENIZER_RECONSTRUCTION_VIDEO_PATH),
        backend_args={
            "backend": "s3",
            "s3_credential_path": UNITTEST_CONFIG.object_store_credential_data,
        },
    )[0][0]  # First frame: (H, W, C) in [0, 255]

    H, W, C = video_in_numpy.shape
    print(f"\n[UniAE S3 Image] Original image shape: ({H}, {W}, {C})")

    # Pad spatial dimensions to be divisible by 16
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        video_in_numpy = np.pad(video_in_numpy, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    H_padded, W_padded = video_in_numpy.shape[:2]

    # Convert to tensor [-1, 1] as single image (encode handles repeat internally)
    image_tensor = torch.from_numpy(video_in_numpy).float().permute(2, 0, 1) / 127.5 - 1.0  # (C, H, W)
    image_batch = image_tensor.unsqueeze(0).cuda()  # (1, C, H, W)

    print(f"[UniAE S3 Image] Input tensor shape: {image_batch.shape}")

    # Encode and decode (encode handles repeat to 4 frames internally)
    latents = vae.encode(image_batch)
    print(f"[UniAE S3 Image] Latent shape: {latents.shape}")
    print(f"[UniAE S3 Image] Latent statistics: mean={latents.mean():.4f}, std={latents.std():.4f}")
    video_recon = vae.decode(latents)
    print(f"[UniAE S3 Image] Reconstructed shape: {video_recon.shape}")

    # Take the first frame as reconstructed image
    recon_image = video_recon[0, :, 0].clamp(-1, 1)  # (C, H, W)
    recon_numpy = ((recon_image.float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

    # Crop back to original size
    recon_numpy = recon_numpy[:H, :W]

    # Compute PSNR against original (unpadded)
    gt = video_in_numpy[:H, :W].astype(np.float32)
    recon_f = recon_numpy.astype(np.float32)
    mse = np.mean((gt - recon_f) ** 2)
    psnr = 10 * np.log10(255**2 / max(mse, 1e-10))
    print(f"[UniAE S3 Image] PSNR: {psnr:.2f} dB")

    # Save original and reconstruction side by side
    output_path = os.path.expanduser("logs/uniae_image_recon.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    orig_img = Image.fromarray(video_in_numpy[:H, :W])
    recon_img = Image.fromarray(recon_numpy)
    side_by_side = Image.new("RGB", (W * 2 + 10, H))
    side_by_side.paste(orig_img, (0, 0))
    side_by_side.paste(recon_img, (W + 10, 0))
    side_by_side.save(output_path)
    print(f"[UniAE S3 Image] Saved side-by-side to: {output_path}")
