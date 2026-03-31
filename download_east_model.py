from __future__ import annotations
import urllib.request
import config
EAST_URL = 'https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV/raw/master/frozen_east_text_detection.pb'

def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if config.EAST_MODEL_PATH.is_file():
        print(f'Already present: {config.EAST_MODEL_PATH}')
        return
    print(f'Downloading EAST model to {config.EAST_MODEL_PATH} ...')
    urllib.request.urlretrieve(EAST_URL, config.EAST_MODEL_PATH)
    print('Done.')
if __name__ == '__main__':
    main()
