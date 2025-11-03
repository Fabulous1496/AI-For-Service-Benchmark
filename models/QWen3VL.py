# models/QWen3VL.py
import torch
from typing import List
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from utils.chunk_and_sample_videos import sample_video_segments_torchcodec

class EvalQWen3VL:
    def __init__(self, args, device: str = "cuda", torch_dtype: str = "auto", max_new_tokens: int = 256):
        self.args = args
        self.device = device
        self.max_new_tokens = max_new_tokens

        model_kwargs = {
            "dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if getattr(args, "attn_implementation", None) is not None:
            model_kwargs["attn_implementation"] = getattr(args, "attn_implementation")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            **model_kwargs
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True
        )

    def inference(self, video_path: str, prompt: str) -> List[str]:
        segments = sample_video_segments_torchcodec(
            video_path,
            segment_seconds=getattr(self.args, "segment_seconds", 300),
            nframes_per_segment=getattr(self.args, "nframes_per_segment", 900),
            device=torch.device("cpu"),
            max_pixels=getattr(self.args, "max_pixels", None)
        )

        results: List[str] = []
        model_device = next(self.model.parameters()).device

        for seg_tensor in segments:
            thwc = seg_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": "<video>"},
                    {"type": "text", "text": prompt},
                ],
            }]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=None,
                videos=[thwc],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model_device)
            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            input_ids = inputs.input_ids
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, gen)]
            decoded = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            results.extend(decoded)

        return results
