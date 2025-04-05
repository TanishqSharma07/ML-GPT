from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import torch

from peft import PeftModel

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
)

checkpoint = "Hunter700/phi2-ml-qa-qlora"
device = "cuda" if torch.cuda.is_available() else "cpu"  # "cuda" or "cpu"

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    device_map='auto',
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True

).to(device)

model = PeftModel.from_pretrained(model, checkpoint)
model.eval()

def predict(message, history):
    history.append({"role": "user", "content": message})
    input_text = f"### Question: {message}\n ### Answer:" #tokenizer.apply_chat_template(history, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)  
    outputs = model.generate(inputs, max_new_tokens=100, temperature=0.2, top_p=0.9, do_sample=True)
    decoded = tokenizer.decode(outputs[0])
    response = decoded.split("### Answer:")[1]
    return response

demo = gr.ChatInterface(predict, type="messages", title = "ML GPT", examples = ["What is confusion matrix?", "What is test train split?"])

demo.launch()