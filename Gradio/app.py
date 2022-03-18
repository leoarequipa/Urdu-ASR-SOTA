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
    return prediction


################### Gradio Web APP ################################

title = "Urdu Automatic Speech Recognition"

description = """
<p>
<center>
Savta Depth is a collaborative Open Source Data Science project for monocular depth estimation - Turn 2d photos into 3d photos. To test the model and code please check out the link bellow.
<img src="https://huggingface.co/kingabzpro/wav2vec2-large-xls-r-300m-Urdu/resolve/main/Images/cover.jpg" alt="logo" width="250"/>
</center>
</p>
"""
article = "<p style='text-align: center'><a href='https://dagshub.com/OperationSavta/SavtaDepth' target='_blank'>SavtaDepth Project from OperationSavta</a></p><p style='text-align: center'><a href='https://colab.research.google.com/drive/1XU4DgQ217_hUMU1dllppeQNw3pTRlHy1?usp=sharing' target='_blank'>Google Colab Demo</a></p></center></p>"

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
        # description=description,
        article=article,
        examples=examples,
        theme="peach",
    )

    iface.launch(enable_queue=True)


# enable_queue=True,auth=("admin", "pass1234")

if __name__ == "__main__":
    main()

