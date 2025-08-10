import os
from io import BytesIO
from typing import List, Dict, Union
from pathlib import Path

import fitz
import pytesseract
from PIL import Image
import pandas as pd

from transformers import BlipProcessor, BlipForConditionalGeneration

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

class DocPreprocessor:
    def __init__(self, ocr_lang='chi_sim'):
        self.ocr_lang = ocr_lang
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        doc = fitz.open(pdf_path)
        texts = []
        for page_index, page in enumerate(doc):
            print(page_index)
            texts.append(page.get_text())
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                ocr_text = self.perform_ocr_on_image(image)
                caption = self.generate_caption(image)
                texts.append(ocr_text)
                texts.append(caption)
        return texts

    def extract_images_from_pdf(self, pdf_path: str) -> List[Image.Image]:
        images = []
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_index)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                images.append(image)
        return images

    def perform_ocr_on_image(self, image: Image.Image) -> str:
        #tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'
        return pytesseract.image_to_string(image, lang=self.ocr_lang)

    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.blip_processor(images=image, return_tensors="pt")
        out = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def extract_text_from_excel(self, excel_path: str) -> List[str]:
        df = pd.read_excel(excel_path, sheet_name=None)
        texts = []
        for sheet_name, sheet in df.items():
            texts.extend(sheet.astype(str).values.flatten().tolist())
        return texts

    def extract_text_from_txt(self, txt_path: str) -> List[str]:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.readlines()

    def process_documents(self, file_paths: List[str]) -> List[str]:
        all_texts = []
        for path in file_paths:
            ext = Path(path).suffix.lower()
            if ext == '.pdf':
                all_texts.extend(self.extract_text_from_pdf(path))
            elif ext in ['.xlsx', '.xls']:
                all_texts.extend(self.extract_text_from_excel(path))
            elif ext == '.txt':
                all_texts.extend(self.extract_text_from_txt(path))
        return all_texts
