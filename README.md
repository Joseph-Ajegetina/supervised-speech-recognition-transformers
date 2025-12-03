# Supervised Speech Recognition with Transformers

This project delves into the application of Transformer models for Automatic Speech Recognition (ASR), with a dual focus on foundational understanding and practical application. It addresses a real-world problem—medical speech transcription—by leveraging this powerful architecture.


## Project Structure
-   `transformer_educational.ipynb`: A Jupyter notebook providing an educational walkthrough of the Transformer architecture, implemented from scratch using PyTorch. It demonstrates building a GPT-style decoder-only model for character-level language modeling on the Tiny Shakespeare dataset.

- `pretrained_whisper_finetuning.ipynb`: A Jupyter notebook detailing the process of fine-tuning the OpenAI Whisper Large V2 model for Automatic Speech Recognition on the AfriSpeech-200 dataset. It covers data loading, preprocessing, model configuration, training, and evaluation, including a Gradio demo.



## Part I: Building a Transformer from Scratch

This section focuses on gaining a fundamental understanding of the Transformer architecture by implementing a decoder-only model from the ground up.

### Objective and Task

The primary objective is to build a generative language model similar to GPT. The model is trained on the Tiny Shakespeare dataset at the character level, aiming to generate text that mimics Shakespeare's distinctive style. This process illuminates the mechanics of self-attention, multi-head attention, feed-forward networks, and positional embeddings.

### Implementation Details

The implementation, detailed in `transformer_educational.ipynb`, is built in PyTorch and covers:
-   **Self-Attention**: The core mechanism allowing the model to weigh the importance of different tokens, with masked self-attention for autoregressive generation.
-   **Multi-Head Attention**: Running multiple attention mechanisms in parallel for diverse perspectives.
-   **Feed-Forward Network**: Position-wise computation after attention.
-   **Transformer Block**: Comprising Multi-Head Attention and a Feed-Forward Network, incorporating Layer Normalization (Pre-Norm formulation) and Residual Connections for training stability.
-   **GPTLanguageModel**: The full model stacking these blocks, along with token and positional embeddings, and a final linear layer for vocabulary projection.

## Part II: Fine-Tuning a Pretrained Transformer for ASR

This part of the project applies transfer learning to solve a practical ASR task.

### Objective and Motivation

Motivated by a request from Ayamra, a network of African hospitals, the goal is to build a medical speech recognition system to reduce the administrative burden on doctors. This requires high accuracy on specialized medical terminology spoken with a variety of Pan-African accents. By fine-tuning OpenAI's Whisper model, we adapt its general speech recognition capabilities to this specific domain.

### Task and Methodology

The complete methodology is implemented in `pretrained_whisper_finetuning.ipynb`.
-   **Dataset**: The `tobiolatunji/afrispeech-200` dataset, consisting of approximately 200 hours of English speech from African speakers, is used.
-   **Model**: `openai/whisper-small` is selected as the base pretrained model due to its state-of-the-art performance.
-   **Preprocessing**: Audio data is loaded in streaming mode, resampled to 16kHz, and converted to log-Mel spectrograms using `WhisperFeatureExtractor`. Transcripts are tokenized using `WhisperTokenizer`.
-   **Fine-Tuning**: The Hugging Face `Seq2SeqTrainer` is employed for fine-tuning, utilizing a batch size of 16, gradient accumulation, and `fp16` (mixed-precision training) for efficient adaptation.

### Evaluation and Results

The model's performance is evaluated using the **Word Error Rate (WER)**. After fine-tuning, the model achieved a WER of **14.77%** on the validation set and **16.14%**, demonstrating significant accuracy for this challenging, low-resource domain. A Gradio demo is included in the notebook for live testing of the fine-tuned model.
