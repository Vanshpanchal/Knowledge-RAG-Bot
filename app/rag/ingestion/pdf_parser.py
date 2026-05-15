from pypdf import PdfReader


class PDFParser:

    @staticmethod
    def extract_text(file_path: str) -> dict:
        reader = PdfReader(file_path)

        extracted_pages = []

        for index, page in enumerate(reader.pages):
            text = page.extract_text()

            extracted_pages.append({"page": index + 1, "text": text if text else ""})

        full_text = "\n".join(str(page["text"]) for page in extracted_pages)

        return {
            "page_count": len(reader.pages),
            "pages": extracted_pages,
            "full_text": full_text,
        }
