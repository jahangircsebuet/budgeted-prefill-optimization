# 2. segmentation.py (document segmentation)
import spacy
import nltk
nltk.download("punkt")

# nlp = spacy.load("en_core_web_sm")

# def segment_text(raw_text: str):
#     """Split text into sentences or paragraphs."""
#     doc = nlp(raw_text)
#     segments = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
#     return segments

# alternate to spacy for simplicity
import nltk
nltk.download("punkt")

def segment_text(raw_text: str):
    """Segment text into sentences using NLTK."""
    return nltk.sent_tokenize(raw_text)