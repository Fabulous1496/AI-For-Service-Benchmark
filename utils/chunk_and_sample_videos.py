import torch
import torch.nn.functional as F
import numpy as np
from torchcodec.decoders import VideoDecoder

"""
把视频按 segment_seconds 分段，每段均匀采样 nframes_per_segment 帧。
返回：list，每个元素为 torch.Tensor shape [T, C, H, W] dtype uint8，T≈nframes_per_segment。
如果 max_pixels 不为 None，会对 HxW 进行缩放，使得 H*W <= max_pixels。
"""

def sample_video_segments_torchcodec(
    video_path: str,
    segment_seconds: int = 300,
    nframes_per_segment: int = 900,
    device: torch.device = torch.device("cpu"),
    max_pixels: int = None
) -> list:
    
    decoder = VideoDecoder(video_path, device=device, dimension_order="NCHW")
    total_frames = len(decoder)
    metadata = decoder.metadata
    fps = metadata.average_fps_from_header if hasattr(metadata, "average_fps_from_header") else (total_frames / metadata.duration_seconds)
    total_seconds = metadata.duration_seconds

    segments = []
    num_segments = int(np.ceil(total_seconds / segment_seconds))

    for seg_idx in range(num_segments):
        start_sec = seg_idx * segment_seconds
        end_sec = min((seg_idx + 1) * segment_seconds, total_seconds)
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps) if (end_sec * fps) < total_frames else total_frames - 1
        if end_frame <= start_frame:
            continue

        # 均匀抽帧索引
        indices = torch.linspace(start_frame, end_frame, steps=nframes_per_segment).round().long().tolist()
        frame_batch = decoder.get_frames_at(indices=indices)  # FrameBatch
        frames_tensor = frame_batch.data  # shape [T, C, H, W], dtype uint8

        if max_pixels is not None:
            T, C, H, W = frames_tensor.shape
            scale_factor = (max_pixels / (H * W)) ** 0.5
            if scale_factor < 1.0:
                new_H = int(H * scale_factor)
                new_W = int(W * scale_factor)
                # 使用双线性插值缩放
                frames_tensor = F.interpolate(
                    frames_tensor.float(), size=(new_H, new_W), mode='bilinear', align_corners=False
                ).to(dtype=torch.uint8)

        segments.append(frames_tensor)

    return segments
