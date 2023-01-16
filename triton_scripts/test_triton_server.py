import asyncio
import sys

import cv2
import numpy as np

sys.path.append("..")

import triton_client
from app import settings
from app.detector import Detector

DETECTOR = triton_client.Client(settings.TRITON_URL, settings.MODEL_NAME)

async def test(image_cv2):
    result, time = await DETECTOR.predict(image_cv2, 'dummy_id')
    return result, time


if __name__ == '__main__':
    test_image = settings.TEST_IMAGE
    image_cv2 = cv2.imread(test_image)
    detector = Detector(settings.WEIGHTS_PT)
    pt_result = detector.detect(image_cv2)

    loop = asyncio.get_event_loop()
    triton_result, time = loop.run_until_complete(test(image_cv2))
    print('Elapsed time:', time)

    # compare outputs with torch and triton models
    np.testing.assert_allclose(pt_result.cpu().numpy(), triton_result.cpu().numpy(), rtol=1e-03, atol=1e-04)

    print("-" * 42)
    print("Well done! Triton model result is equal to pytorch model result.")
