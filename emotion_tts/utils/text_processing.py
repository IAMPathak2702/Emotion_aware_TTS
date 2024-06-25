import re

def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def text_to_sequence(text, cleaner_names):
    # This is a placeholder. You should implement or use a proper text-to-sequence converter
    # that matches your model's expectations.
    sequence = [ord(c) for c in clean_text(text)]
    return sequence