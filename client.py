import base64
import json
from io import BytesIO
from pathlib import Path
from xmlrpc.client import ServerProxy, Transport

import cv2
from PIL import Image


DEFAULT_DETECTOR_URL = "http://127.0.0.1:6789"


class TimeoutTransport(Transport):
    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = self.timeout
        return conn


class RexOmniClient:
    def __init__(self, base_url=DEFAULT_DETECTOR_URL, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = ServerProxy(
            self.base_url,
            allow_none=True,
            transport=TimeoutTransport(timeout=self.timeout),
        )

    @staticmethod
    def _encode_bgr_to_rgb_base64_jpeg(image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        bio = BytesIO()
        Image.fromarray(rgb).save(bio, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(bio.getvalue()).decode('ascii')}"

    def detect(self, image, target):
        payload = {"img": self._encode_bgr_to_rgb_base64_jpeg(image), "cats": [target]}
        body = json.loads(self.client.infer(json.dumps(payload)))
        if not body.get("ok", False):
            raise RuntimeError(body.get("error", "detector rpc error"))
        data = body.get("result", {})
        det_result = []
        for ins in data.get(target, []):
            coords = ins.get("coords", [])
            if len(coords) < 4:
                continue
            det_result.append([int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]), 1.0])
        return det_result


def main():
    image_path = Path("PATH")
    target = "TARGET"

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    client = RexOmniClient()
    det_result = client.detect(image, target)

    vis_image = image.copy()
    for x0, y0, x1, y1, score in det_result:
        cv2.rectangle(vis_image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            vis_image,
            f"{target} {score:.2f}",
            (x0, max(30, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("RexOmni Detection", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
