import argparse
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from pathlib import Path


def set_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, help="Path to the image")
    return parser


def greyscale(img):
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return greyscale


def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen


def sepia(img):
    img_sepia = np.array(img, dtype=np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168],[0.393, 0.769, 0.189]]))
    img_sepia[np.where(img_sepia > 255)] = 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia


def HDR(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def pencil_sketch_grey(img):
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_gray, sk_color


def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    summer= cv2.merge((blue_channel, green_channel, red_channel ))
    return summer


def Winter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    return win


def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges


def cartoonify(image):
    line_size = 7
    blur_value = 7
    edges = edge_mask(image, line_size, blur_value)
    dst = cv2.edgePreservingFilter(image, flags=2, sigma_s=64, sigma_r=0.25)
    cartoon = cv2.bitwise_and(dst, dst, mask=edges)
    return cartoon


def process_image(image_path: Path, out_path: Path) -> None:
    image = cv2.imread(str(image_path))

    grey = greyscale(image)
    cv2.imwrite(str(out_path / "greyscale.png"), grey)

    sharp = sharpen(image)
    cv2.imwrite(str(out_path / "sharpen.png"), sharp)

    sep = sepia(image)
    cv2.imwrite(str(out_path / "sepia.png"), sep)

    hdr = HDR(image)
    cv2.imwrite(str(out_path / "detailed.png"), hdr)

    grey_sketch, color_sketch = pencil_sketch_grey(image)
    cv2.imwrite(str(out_path / "pencil_color.png"), color_sketch)
    cv2.imwrite(str(out_path / "pencil_grey.png"), grey_sketch)

    cv2.imwrite(str(out_path / "summer.png"), Summer(image))

    cv2.imwrite(str(out_path / "winter.png"), Winter(image))

    stylized = cv2.stylization(image, sigma_s=20, sigma_r=0.5)
    cv2.imwrite(str(out_path / "stylized.png"), stylized)

    cartoon = cartoonify(image)
    cv2.imwrite(str(out_path / "cartoon.png"), cartoon)


if __name__ == "__main__":
    arg_parser = set_args()
    args = arg_parser.parse_args()
    image_path = Path(args.image_path)
    image_name = image_path.stem
    out_path = Path("./" + f"{image_name}_filtered")
    out_path.mkdir(exist_ok=True)
    
    process_image(image_path, out_path)
