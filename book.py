import PyPDF2
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import nltk
import re



nltk.download('punkt')


def extract_text_from_pdf(pdf_path, start_page=15, end_page=467):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(start_page-1, end_page):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + " "
    return text


def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text

def summarize_text(text, num_words, model_name='google/pegasus-xsum'):
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    tokens = tokenizer(text, truncation=True, padding='longest', return_tensors='pt')
    
    summary_ids = model.generate(tokens.input_ids, max_length=num_words, num_beams=4, length_penalty=2.0, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def summarize_pdf_book(pdf_path, num_words, start_page=15, end_page=467):
    extracted_text = extract_text_from_pdf(pdf_path, start_page, end_page)
    
    cleaned_text = preprocess_text(extracted_text)
    
    summary = summarize_text(cleaned_text, num_words)
    
    return summary






pdf_path = 'book.pdf'
num_words = 3000  
summary = summarize_pdf_book(pdf_path, num_words)
print(summary)
