# import pandas as pd
# from pycaret.classification import load_model, predict_model
# model = load_model('sentiment_model')
# def predict_sentiment(text):
#     data = pd.DataFrame({'text': [text]})
#     data['text_length'] = data['text'].str.len()
#     result = predict_model(model, data=data)
#     return result['prediction_label'][0]

import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = 'indobert_sentiment'
MAX_LEN = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model = model.to(device)
model.eval()

label_map = {0: 'negatif', 1: 'positif'}

def predict_sentiment(text):
    encoding = tokenizer(
        text.lower(),
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1)
        pred    = torch.argmax(probs, dim=1).item()
        conf    = probs[0][pred].item()

    return {label_map[pred]: conf, label_map[1 - pred]: 1 - conf}

examples = [
    ["IKN adalah proyek yang sangat bagus dan menjanjikan untuk masa depan Indonesia"],
    ["IKN hanya proyek koruptor dan bentuk nepotisme yang merugikan rakyat"],
    ["Pembangunan IKN berjalan dengan baik dan lancar sesuai target"],
    ["Anggaran IKN terlalu besar dan tidak tepat sasaran"],
    ["Semoga IKN bisa selesai dan membawa manfaat bagi seluruh warga Indonesia"],
]

with gr.Blocks(theme=gr.themes.Soft(), title="Sentiment Analysis IKN") as iface:
    gr.Markdown(
        """
        # Sentiment Analysis IKN
        Analisis sentimen opini publik terhadap **Ibu Kota Nusantara (IKN)** menggunakan model **IndoBERT**
        yang telah di-*fine-tune* pada data Bahasa Indonesia.

        > Model: `indobenchmark/indobert-base-p1` (fine-tuned) &nbsp;|&nbsp; Test Accuracy: **89.59%**
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                lines=4,
                placeholder="Ketik atau tempel opini kamu tentang IKN di sini...",
                label="Teks Opini",
                show_copy_button=True,
            )
            with gr.Row():
                clear_btn  = gr.Button("Hapus", variant="secondary")
                submit_btn = gr.Button("Analisis Sentimen", variant="primary")

        with gr.Column(scale=2):
            label_output = gr.Label(
                label="Hasil Sentimen",
                num_top_classes=2,
            )

    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="Contoh Teks",
    )

    gr.Markdown(
        """
        ---
        <div style='text-align:center; color:gray; font-size:0.85em'>
        Dibuat untuk keperluan riset analisis opini publik IKN &nbsp;·&nbsp; PBA 2026
        </div>
        """
    )

    submit_btn.click(fn=predict_sentiment, inputs=text_input, outputs=label_output)
    clear_btn.click(fn=lambda: ("", None), outputs=[text_input, label_output])

iface.launch()
