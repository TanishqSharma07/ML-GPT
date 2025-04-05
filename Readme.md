# 🧠 ML Chatbot – Fine-Tuned LLM
This project is a chatbot powered by a fine-tuned LLM, specifically designed to answer Machine Learning questions. The model is deployed using Gradio, allowing easy interaction through a web interface.

# 🚀 Features
- Fine-tuned on a curated dataset of ML questions.

- Built on top of the Phi-2 model (~3B parameters).

- Lightweight and efficient thanks to PEFT techniques.

- Simple Gradio UI (app.py) for local or web-based usage.

# 🛠️ Fine-Tuning Details
1. Due to the large size of the Phi-2 model, full fine-tuning was not feasible. Instead, the following Parameter-Efficient Fine-Tuning (PEFT) methods were used:

2. Quantization: The base model was quantized to reduce memory usage and training cost.

3. LoRA (Low-Rank Adaptation): LoRA adapters were applied to fine-tune only a subset of model parameters efficiently.

4. This approach ensures performance gains while keeping the resource footprint minimal.
