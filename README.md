# Supervised Speech Recognition with Transformers

This project delves into the application of Transformer models for Automatic Speech Recognition (ASR), with a dual focus on foundational understanding and practical application. It addresses a real-world problem—medical speech transcription—by leveraging this powerful architecture.


## Project Structure
-   `transformer_educational.ipynb`: A Jupyter notebook providing an educational walkthrough of the Transformer architecture, implemented from scratch using PyTorch. It demonstrates building a GPT-style decoder-only model for character-level language modeling on the Tiny Shakespeare dataset.

- `pretrained_whisper_finetuning.ipynb`: A Jupyter notebook detailing the process of fine-tuning the OpenAI Whisper small model for Automatic Speech Recognition on the AfriSpeech-200 dataset. It covers data loading, preprocessing, model configuration, training, and evaluation, including a Gradio demo.


### Evaluation and Results

The model's performance is evaluated using the **Word Error Rate (WER)**. After fine-tuning, the model achieved a WER of **14.77%** on the validation set and **16.14%**, demonstrating significant accuracy for this challenging, low-resource domain. A Gradio demo is included in the notebook for live testing of the fine-tuned model.
