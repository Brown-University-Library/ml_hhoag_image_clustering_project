# /// script
# requires-python = "~=3.12"
# dependencies = [
#     "pillow",
# ]
# ///

from PIL import Image


def convert_jp2_to_jpeg(input_path, output_path):
    with Image.open(input_path) as img:
        # Convert image to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(output_path, 'JPEG')


if __name__ == '__main__':
    input_path = '/Users/birkin/Desktop/hh_images/HH018977_0030.jp2'
    output_path = '/Users/birkin/Desktop/HH018977_0030.jpg'
    convert_jp2_to_jpeg(input_path, output_path)
