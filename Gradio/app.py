import os
from datasets import load_dataset, Audio
from transformers import pipeline
import gradio as gr

############### HF ###########################

HF_TOKEN = os.getenv("HF_TOKEN")

hf_writer = gr.HuggingFaceDatasetSaver(HF_TOKEN, "Urdu-ASR-flags")

############## DVC ################################

Model = "Model"

if os.path.isdir(".dvc"):
    print("Running DVC")
    # os.system("dvc config cache.type copy")
    # os.system("dvc config core.no_scm true")
    if os.system(f"dvc pull {Model} -r origin") != 0:
        exit("dvc pull failed")
    # os.system("rm -r .dvc")
# .apt/usr/lib/dvc

############## Inference ##############################


def asr(audio):

    asr = pipeline("automatic-speech-recognition", model=Model)
    prediction = asr(audio, chunk_length_s=5, stride_length_s=1)
    return prediction["text"]


################### Gradio Web APP ################################

title = "Urdu Automatic Speech Recognition"

description = """
<p>
<center>
This model is a fine-tuned version of facebook/wav2vec2-xls-r-300m on the common_voice dataset.
</center>
</p>
<center>
<img src="https://huggingface.co/spaces/kingabzpro/Urdu-ASR-SOTA/resolve/main/Images/cover.jpg" alt="logo" width="550"/>
</center>
"""

article = "<p style='text-align: center'><a href='https://dagshub.com/kingabzpro/Urdu-ASR-SOTA' target='_blank'>Source Code on DagsHub</a></p><p style='text-align: center'><a href='https://huggingface.co/blog/fine-tune-xlsr-wav2vec2' target='_blank'>Fine-tuning XLS-R for Multi-Lingual ASR with 🤗 Transformers</a></p></center></p>"

examples = [["Sample/sample1.mp3"], ["Sample/sample2.mp3"], ["Sample/sample3.mp3"]]


Input = gr.inputs.Audio(
    source="microphone",
    type="filepath",
    optional=True,
    label="Please Record Your Voice",
)
Output = gr.outputs.Textbox(label="Urdu Script")


def main():
    iface = gr.Interface(
        asr,
        Input,
        Output,
        title=title,
        flagging_options=["incorrect", "worst", "ambiguous"],
        allow_flagging="manual",
        flagging_callback=hf_writer,
        description=description,
        article=article,
        examples=examples,
        theme="grass",
    )

    iface.launch(enable_queue=True)


# enable_queue=True,auth=("admin", "pass1234")

if __name__ == "__main__":
    main()

