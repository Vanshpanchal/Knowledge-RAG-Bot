import re


class TextCleaner:

    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r"\n+", "\n", text)

        text = re.sub(r"\s+", " ", text)

        text = text.strip()

        return text
