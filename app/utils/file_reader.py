from io import BytesIO
from pypdf import PdfReader
from docx import Document


def read_resume_from_upload(file):
    name = file.filename.lower()

    # Always reset pointer
    file.file.seek(0)

    if name.endswith(".pdf"):
        reader = PdfReader(file.file)
        return " ".join([p.extract_text() or "" for p in reader.pages])

    elif name.endswith(".docx"):
        # Convert SpooledTemporaryFile → BytesIO
        data = file.file.read()
        doc = Document(BytesIO(data))

        return "\n".join(p.text for p in doc.paragraphs if p.text)

    else:
        return file.file.read().decode("utf-8", errors="ignore")
