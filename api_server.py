import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
import base64
from io import BytesIO
from xmlrpc.server import SimpleXMLRPCServer

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
REX_OMNI_DIR = ROOT_DIR / "Rex-Omni"
sys.path.insert(0, str(REX_OMNI_DIR))

from rex_omni import RexOmniWrapper

torch.set_grad_enabled(False)

class perception_api:
    def __init__(self, model_path="./models", gpu_memory_utilization=0.3):
        self.model_path = str((ROOT_DIR / "models").resolve()) if model_path == "./models" else model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.model = RexOmniWrapper(
            model_path=self.model_path,
            backend="vllm",  # or "vllm" for faster inference
            max_tokens=4096,
            temperature=0.0,
            top_p=0.05,
            top_k=1,
            repetition_penalty=1.05,
            gpu_memory_utilization=self.gpu_memory_utilization
        )

    def infer(self, param):
        param = json.loads(param)
        img = param["img"]
        cats = param["cats"]
        assert img.startswith("data:image/jpeg;base64")
        img = img.split(",")[1]
        img = base64.b64decode(img)
        img = BytesIO(img)
        img = Image.open(img)
        result = self.model.inference(images=img, task="detection", categories=cats)[0]
        if result["success"]:
            return json.dumps(dict(ok=True, result=result["extracted_predictions"]))
        return json.dumps(dict(ok=False, result=[]))

if __name__ == "__main__":
    # try:
    #     mp.set_start_method("spawn")
    # except RuntimeError:
    #     if mp.get_start_method(allow_none=True) != "spawn":
    #         raise
    port = 6789
    controller = perception_api()
    server = SimpleXMLRPCServer(("0.0.0.0", port))
    server.register_instance(controller)
    server.serve_forever()
