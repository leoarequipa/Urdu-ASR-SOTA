# Urdu-ASR-SOTA

Automatic Speech Recognition using Facebook wav2vec2-xls-r-300m model and mozilla-foundation common_voice_8_0 Urdu Dataset.

## wav2vec2-large-xls-r-300m-Urdu
This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the common_voice dataset.

It achieves the following results on the evaluation set:
- Loss: 0.9889
- Wer: 0.5607
- Cer: 0.2370

#### Evaluation Commands
To evaluate on `mozilla-foundation/common_voice_8_0` with split `test`

```bash
python3 ./eval.py --model_id ./Model --dataset ./Data --config ur --split test --chunk_length_s 5.0 --stride_length_s 1.0 --log_outputs
```

```python
import torch
from datasets import load_dataset, Audio
from transformers import pipeline
import torchaudio.functional as F
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


### Eval results on Common Voice 8 "test" (WER):

| Without LM | With LM (run `./eval.py`) |
|---|---|
| 56.21 | 46.37 |