# Urdu-ASR-SOTA

Automatic Speech Recognition using Facebook wav2vec2-xls-r-300m model and mozilla-foundation common_voice_8_0 Urdu Dataset.

## Model Finetunning
This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the common_voice dataset.

It achieves the following results on the evaluation set:
- Loss: 0.9889
- Wer: 0.5607
- Cer: 0.2370

## Quick Prediction

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
## Eval Results

| Without LM | With LM|
|---|---|
| 56.21 | 46.37 |


## SOTA
- [x] Add Language Model
- [] Webapp/API
- [] Denoise Audio
- [] Text Processing
- [] Spelling Mistakes
- [x] Hyperparameters optimization 
- [] Training on 300 Epochs & 64 Batch Size
- [] Improved Language Model
- [] Contribute to Urdu ASR Audio Dataset

