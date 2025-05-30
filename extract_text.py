# extract_text.py
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    sample_path = "data/input/sample.pdf"
    output_text = extract_text_from_pdf(sample_path)
    with open("data/extracted/output_text.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    print("Text extracted and saved.")
