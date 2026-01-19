from pypdf import PdfReader

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        text.append(t)
    return "\n".join(text)

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    text = text.replace("\r", "\n")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(words):
            break
    return chunks

