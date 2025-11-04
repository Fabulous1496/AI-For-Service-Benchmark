import torch
import torch.nn.functional as F
import numpy as np
from torchcodec.decoders import VideoDecoder

def _resize_thwc_np(arr_thwc: np.ndarray, max_pixels: int) -> np.ndarray:
    if max_pixels is None or max_pixels <= 0:
        return arr_thwc
    T, H, W, C = arr_thwc.shape
    if H * W <= max_pixels:
        return arr_thwc
    scale = (max_pixels / (H * W)) ** 0.5
    newH = max(32, int(H * scale))
    newW = max(32, int(W * scale))
    
    # 简单最近邻缩放
    out = np.empty((T, newH, newW, C), dtype=np.uint8)
    ys = (np.linspace(0, H - 1, newH)).astype(np.int64)
    xs = (np.linspace(0, W - 1, newW)).astype(np.int64)
    for t in range(T):
        out[t] = arr_thwc[t][np.ix_(ys, xs)]
    return out


def sample_video_segments_torchcodec(
    video_path: str,
    segment_seconds: int = 600,
    nframes_per_segment: int = 1500,
    device: torch.device = torch.device("cpu"),
    max_pixels: int = 360*480
) -> list:
    
    decoder = VideoDecoder(video_path, device=device, dimension_order="NCHW")
    total_frames = len(decoder)
    metadata = decoder.metadata
    
    if hasattr(metadata, "average_fps_from_header") and metadata.average_fps_from_header > 0:
         fps = metadata.average_fps_from_header
    elif metadata.duration_seconds > 0:
        fps = total_frames / metadata.duration_seconds
    else:
        fps = 30.0 # 默认回退
        
    total_seconds = metadata.duration_seconds if metadata.duration_seconds > 0 else (total_frames / fps)

    segments = []
    num_segments = int(np.ceil(total_seconds / segment_seconds))

    for seg_idx in range(num_segments):
        start_sec = seg_idx * segment_seconds
        end_sec = min((seg_idx + 1) * segment_seconds, total_seconds)
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps) if (end_sec * fps) < total_frames else total_frames - 1
        
        if end_frame <= start_frame:
            continue

        current_nframes = min(nframes_per_segment, end_frame - start_frame + 1)
        if current_nframes <= 0:
            continue
            
        indices = torch.linspace(start_frame, end_frame, steps=current_nframes).round().long().tolist()
        
        frame_batch = decoder.get_frames_at(indices=indices)
        frames_tensor = frame_batch.data  # shape [T, C, H, W], dtype uint8

        if max_pixels is not None:
            # TCHW (Tensor, device) -> THWC (NumPy, cpu)
            arr_thwc = frames_tensor.cpu().numpy().transpose(0, 2, 3, 1)
            arr_thwc_resized = _resize_thwc_np(arr_thwc, max_pixels)
            # THWC (NumPy, cpu) -> TCHW (Tensor, device)
            frames_tensor = torch.from_numpy(arr_thwc_resized.transpose(0, 3, 1, 2).copy()).to(device)

        segments.append(frames_tensor)

    return segments