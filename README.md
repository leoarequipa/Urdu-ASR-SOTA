# Urdu Automatic Speech Recognition State of the Art Solution

![cover](Images/cover.png)
Automatic Speech Recognition using Facebook's wav2vec2-xls-r-300m model and mozilla-foundation common_voice_8_0 Urdu Dataset.

## Model Finetunning

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the [common_voice dataset](https://commonvoice.mozilla.org/en/datasets).

It achieves the following results on the evaluation set:

- Loss: 0.9889
- Wer: 0.5607
- Cer: 0.2370

## Quick Prediction

Install all dependecies using `requirment.txt` file and then run bellow command to predict the text:

```python
import torch
from datasets import load_dataset, Audio
from transformers import pipeline
model = "Model"
data = load_dataset("Data", "ur", split="test", delimiter="\t")
def path_adjust(batch):
    batch["path"] = "Data/ur/clips/" + str(batch["path"])
    return batch
data = data.map(path_adjust)
sample_iter = iter(data.cast_column("path", Audio(sampling_rate=16_000)))
sample = next(sample_iter)

asr = pipeline("automatic-speech-recognition", model=model)
prediction = asr(
            sample["path"]["array"], chunk_length_s=5, stride_length_s=1)
prediction
# => {'text': 'اب یہ ونگین لمحاتانکھار دلمیں میںفوث کریلیا اجائ'}
```

## Evaluation Commands

To evaluate on `mozilla-foundation/common_voice_8_0` with split `test`, you can copy and past the command to the terminal.

```bash
python3 eval.py --model_id Model --dataset Data --config ur --split test --chunk_length_s 5.0 --stride_length_s 1.0 --log_outputs
```

**OR**
Run the simple shell script

```bash
bash run_eval.sh
```

## Language Model

[Boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram)

- Get suitable Urdu text data for a language model
- Build an n-gram with KenLM
- Combine the n-gram with a fine-tuned Wav2Vec2 checkpoint

Install kenlm and pyctcdecode before running the notebook.

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip pyctcdecode
```

## Eval Results

| Without LM | With LM |
| ---------- | ------- |
| 56.21      | 46.37   |

## Directory Structure

```
<root directory>
    |
    .- README.md
    |
    .- Data/
    |
    .- Model/
    |
    .- Images/
    |
    .- Sample/
    |
    .- Gradio/
    |
    .- Eval Results/
          |
          .- With LM/
          |
          .- Without LM/
          | ...
    .- notebook.ipynb
    |
    .- run_eval.sh
    |
    .- eval.py

```

## Gradio App

![Gradio](Images/gradio.gif)

## SOTA

- [x] Add Language Model
- [x] Webapp/API
- [] Denoise Audio
- [] Text Processing
- [] Spelling Mistakes
- [x] Hyperparameters optimization
- [] Training on 300 Epochs & 64 Batch Size
- [] Improved Language Model
- [] Contribute to Urdu ASR Audio Dataset

## Robust Speech Recognition Challenge 2022

This project was the results of HuggingFace [Robust Speech Recognition Challenge](https://discuss.huggingface.co/t/open-to-the-community-robust-speech-recognition-challenge/13614). I was one of the winner with four state of the art ASR model. Check out my SOTA checkpoints.

- **[Urdu](https://huggingface.co/kingabzpro/wav2vec2-large-xls-r-300m-Urdu)**
- **[Arabic](https://huggingface.co/kingabzpro/wav2vec2-large-xlsr-300-arabic)**
- **[Punjabi](https://huggingface.co/kingabzpro/wav2vec2-large-xlsr-53-punjabi)**
- **[Irish](https://huggingface.co/kingabzpro/wav2vec2-large-xls-r-1b-Irish)**

![winner](Images/winner.png)

## References

- [Common Voice Dataset](https://commonvoice.mozilla.org/en/datasets)
- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
- [Fine-tuning XLS-R for Multi-Lingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)
- [Boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram)
- [HF Model](https://huggingface.co/kingabzpro/wav2vec2-large-xls-r-300m-Urdu)
