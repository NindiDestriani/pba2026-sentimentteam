import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

model = load_model('sentiment_model')

def predict_sentiment(text):
    data = pd.DataFrame({'text': [text]})
    
    # tambahkan fitur yang dibutuhkan model
    data['text_length'] = data['text'].str.len()
    
    result = predict_model(model, data=data)
    return result['prediction_label'][0]

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Masukkan teks..."),
    outputs="text",
    title="Sentiment Analysis IKN",
    description="Model ML menggunakan PyCaret"
)

iface.launch()