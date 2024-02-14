# %%
import os
import re
import textwrap
import warnings
import pyperclip
import soundfile as sf
from pynput import keyboard
from plyer import notification
from playsound import playsound
from num2words import num2words
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

# this is 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.system("export CUDA_VISIBLE_DEVICES=\"\"")

# %% [markdown]
# ** SUMMERIZATION models

# %%
# # facebook/bart-large-cnn
# from transformers import pipeline
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# def facebook_bart_large_cnn(text):
#     summary = summarizer(text, max_length=20 if len(text.split())<=50 else 70, min_length=10, do_sample=False)
#     return summary[0]['summary_text']

# %%
# google/bigbird-pegasus-large-bigpatent
summarytokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-bigpatent")
summarymodel = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-bigpatent", attention_type="original_full")
def google_bigbird_pegasus_large_bigpatent(text):
    inputs = summarytokenizer(text, return_tensors='pt')
    summaryprediction = summarymodel.generate(**inputs)
    thesummary = summarytokenizer.batch_decode(summaryprediction)
    return thesummary[0].replace('<s>', '').replace('</s>', '').strip()

# %% [markdown]
# ** Text To Speech models

# %%
# # microsoft/speecht5_tts
# import torch
# from datasets import load_dataset
# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# def microsoft_speecht5_tts(text):
#     inputs = processor(text=text, return_tensors="pt")
#     return model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder), 22000

# %%
# facebook/fastspeech2-en-ljspeech
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
fairseqmodels, fairseqcfg, fairseqtask = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
fairseqmodel = fairseqmodels[0].to('cpu')
TTSHubInterface.update_cfg_with_data_cfg(fairseqcfg, fairseqtask.data_cfg)
generator = fairseqtask.build_generator(fairseqmodels, fairseqcfg)

def facebook_fastspeech2_en_ljspeech(text):
    sample = TTSHubInterface.get_model_input(fairseqtask, text)
    return TTSHubInterface.get_prediction(fairseqtask, fairseqmodel, generator, sample)

# %%
def show_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        timeout=10
    )

# %%
def summerize(text):
    # print(len(text.split()))
    # print(textwrap.fill(text, 80))

    # return facebook_bart_large_cnn(text)
    return google_bigbird_pegasus_large_bigpatent(text)

# %%
def readoutload(text):
    # wav, rate = microsoft_speecht5_tts(text)
    wav, rate = facebook_fastspeech2_en_ljspeech(text)
    
    sf.write("nowgeneratedspeechforstudy.wav", wav, samplerate=rate)
    show_notification("summary", text)
    playsound("nowgeneratedspeechforstudy.wav")

# %%
def convert_numbers_to_text(text):
    # Regular expression pattern to match numbers
    pattern = r'\b\d+\b'
    
    def replace(match):
        number = int(match.group())
        return num2words(number)
    
    # Replace numbers in the text with their textual representation
    converted_text = re.sub(pattern, replace, text)

    return converted_text

def preprocesstext(text):
    text = text.strip()
    text = text.replace('-\n', '')
    text = text.replace('\n', ' ')
    
    text = convert_numbers_to_text(text)
    # text = text.replace(".", ",").replace("!", ",").replace("?", ",").replace(":", ",").replace(";", ",")
    text = text.replace("(",',').replace(")",',').replace("[",',').replace("]",',').replace("{",',').replace("}",',')
    text = text.replace('"',',').replace("“",',').replace("”",',')
    text = text.replace("-",' ').replace("_",' ').replace("—",' ').replace("–",' ').replace("…",' ')
    
    return text

# %%
def generatebytext(originaltext, mode):
    originaltext = preprocesstext(originaltext)

    if mode == 'stts':
        thesummary = originaltext if len(originaltext.split())<=10 else summerize(originaltext)
        thesummarysplitted = [substr for substr in re.split(r"[.!?;:]", thesummary) if substr]
        print(thesummarysplitted)
        for tmptext in thesummarysplitted: readoutload(tmptext)
    elif mode == 'tts':
        thesummarysplitted = [substr for substr in re.split(r"[.!?;:]", originaltext) if substr]
        print(thesummarysplitted)
        for tmptext in thesummarysplitted: readoutload(tmptext)
