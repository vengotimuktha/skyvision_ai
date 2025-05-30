# extract_images.py
from io import BytesIO
import fitz  # PyMuPDF
import os
from PIL import Image
from pathlib import Path

def extract_images(pdf_path, output_folder="data/extracted/"):
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(BytesIO(image_bytes))
            img_name = f"page{page_num+1}_img{img_index+1}.{image_ext}"
            image.save(os.path.join(output_folder, img_name))
    print("Image extraction complete.")

if __name__ == "__main__":
    extract_images("data/input/sample.pdf")
