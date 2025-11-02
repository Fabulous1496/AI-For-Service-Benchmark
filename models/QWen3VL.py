# models/QWen3VL.py
import torch
from typing import List
from transformers import AutoProcessor, AutoModelForImageTextToText
from utils.chunk_and_sample_videos import sample_video_segments_torchcodec  # 直接导入已有采样函数

class EvalQWen3VL:
    def __init__(self, args, device: str = "cuda", torch_dtype: str = "auto", max_new_tokens: int = 256):
        self.args = args
        self.device = device
        self.max_new_tokens = max_new_tokens

        # 加载模型
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        self.model.eval()

        # 加载 processor
        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True
        )

    def inference(self, video_path: str, prompt: str) -> List[str]:
        """
        对单个视频进行分段采样 + 推理
        返回模型生成的文本列表，每段一个输出
        """
        # 1) 使用外部 chunk_and_sample 模块采样
        segments = sample_video_segments_torchcodec(
            video_path,
            segment_seconds=getattr(self.args, "segment_seconds", 300),
            nframes_per_segment=getattr(self.args, "nframes_per_segment", 900),
            device=torch.device(self.device),
            max_pixels=getattr(self.args, "max_pixels", None)
        )

        results = []

        for seg_tensor in segments:
            # 转 numpy (T,H,W,C) 供 processor 使用
            thwc = seg_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            # 构造消息，Prompt 从外部传入
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": "<video>"},
                    {"type": "text", "text": prompt},
                ],
            }]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.processor(
                text=[text],
                images=None,
                videos=[thwc],
                padding=True,
                return_tensors="pt"
            )

            model_device = next(self.model.parameters()).device
            inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], gen)]
            decoded = self.processor.tokenizer.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            results.extend(decoded)

        return results
