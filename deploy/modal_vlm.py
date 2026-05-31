"""Serverless VLM on Modal — OpenAI-compatible endpoint for hard-case fallback.

Lets a small local box (e.g. RTX 1650) run OCR+KIE while only router-flagged HARD
cases hit this GPU VLM. Scale-to-zero: you pay only while a hard case infers.
Serves /v1/chat/completions (vision) via transformers (the path proven in docai),
so `docai/vlm.py` mode=api plugs in with no change.

Deploy:
    modal deploy deploy/modal_vlm.py
    # -> https://<you>--docai-vlm-serve.modal.run
    export DOCAI_VLM_MODE=api
    export VLM_API_BASE="https://<you>--docai-vlm-serve.modal.run/v1"
    export VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
"""
import modal

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"          # fits L4 (24GB) comfortably
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers==4.49.0", "torch", "torchvision", "accelerate",
                 "qwen-vl-utils", "fastapi[standard]", "pillow")
)
app = modal.App("docai-vlm")
hf_cache = modal.Volume.from_name("docai-hf-cache", create_if_missing=True)


@app.cls(image=image, gpu="L4", volumes={"/root/.cache/huggingface": hf_cache},
         scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=4)
class VLM:
    @modal.enter()
    def load(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        self.processor = AutoProcessor.from_pretrained(MODEL)

    @modal.asgi_app()
    def serve(self):
        import base64, io, re
        from fastapi import FastAPI, Request
        from PIL import Image
        web = FastAPI()

        @web.get("/health")
        def health():
            return {"status": "ok", "model": MODEL}

        @web.post("/v1/chat/completions")
        async def chat(req: Request):
            body = await req.json()
            content = body["messages"][-1]["content"]
            prompt, img = "", None
            for part in content:
                if part.get("type") == "text":
                    prompt = part["text"]
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    b64 = re.sub(r"^data:image/\w+;base64,", "", url)
                    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[img], padding=True,
                                    return_tensors="pt").to("cuda")
            gen = self.model.generate(**inputs, max_new_tokens=body.get("max_tokens", 256),
                                      do_sample=False)
            trimmed = gen[:, inputs.input_ids.shape[1]:]
            out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            return {"choices": [{"message": {"role": "assistant", "content": out}}]}

        return web
