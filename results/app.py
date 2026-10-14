import gradio as gr, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "dmahe09/hausa-news-classifier"   # local folder, or a Hugging Face Hub id such as "your-username/hausa-news-classifier"
MAX_LEN = 256
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

def predict(headline, text):
    enc = tok((headline or "") + ". " + (text or ""), truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, -1)[0]
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}

demo = gr.Interface(fn=predict,
    inputs=[gr.Textbox(label="Headline (Hausa)"), gr.Textbox(label="Article text (Hausa)", lines=8)],
    outputs=gr.Label(num_top_classes=7, label="Predicted topic"),
    title="Hausa News Topic Classifier",
    description="Fine-tuned on MasakhaNEWS (Hausa). Categories: business, entertainment, health, politics, religion, sports, technology.")
demo.launch()
