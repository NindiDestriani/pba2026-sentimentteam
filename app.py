import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

model = load_model('sentiment_model')

def predict_sentiment(text):
    data = pd.DataFrame({'text': [text]})
    data['text_length'] = data['text'].str.len()
    result = predict_model(model, data=data)
    return result['prediction_label'][0]

examples = [
    ["IKN adalah proyek yang sangat bagus dan menjanjikan"],
    ["IKN proyek koruptor dan nepotisme"],
    ["Pembangunan IKN berjalan dengan baik dan lancar"],
]

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Ketik opini kamu tentang IKN di sini..."
    ),
    outputs=gr.Label(label="Hasil Sentimen"),
    examples=examples,
    title="📊 Sentiment Analysis IKN",
    description="Masukkan opini terkait IKN untuk mengetahui sentimen (positif / negatif).",
    theme="soft",
    allow_flagging="never"
)

iface.launch()