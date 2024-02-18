from playsound import playsound
import soundfile as sf

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
# # facebook/fastspeech2-en-ljspeech
# from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
# from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
# fairseqmodels, fairseqcfg, fairseqtask = load_model_ensemble_and_task_from_hf_hub(
#     "facebook/fastspeech2-en-ljspeech",
#     arg_overrides={"vocoder": "hifigan", "fp16": False}
# )
# fairseqmodel = fairseqmodels[0].to('cpu')
# TTSHubInterface.update_cfg_with_data_cfg(fairseqcfg, fairseqtask.data_cfg)
# generator = fairseqtask.build_generator(fairseqmodels, fairseqcfg)

# def facebook_fastspeech2_en_ljspeech(text):
#     sample = TTSHubInterface.get_model_input(fairseqtask, text)
#     return TTSHubInterface.get_prediction(fairseqtask, fairseqmodel, generator, sample)

# %%
# facebook/hf-seamless-m4t-medium
from transformers import AutoProcessor, SeamlessM4TModel
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

def facebook_seamless_streaming(text, src_lang, tgt_lang, spkr_id):
  text_inputs = processor(text, src_lang=src_lang, return_tensors="pt")
  audio_array_from_text = model.generate(**text_inputs, tgt_lang=tgt_lang, spkr_id=spkr_id)[0].cpu().numpy().squeeze()
  return audio_array_from_text

# %%
def readoutload(text, write_address="nowgeneratedspeechforstudy.wav", read=True, src_lang="eng", tgt_lang="eng", spkr_id=0, samplerat=16000):
    # wav, rate = microsoft_speecht5_tts(text)
    wav = facebook_seamless_streaming(text, src_lang, tgt_lang, spkr_id)
    
    sf.write(write_address, wav, samplerate=samplerat)
    if read==True:
        # show_notification("summary", text)
        playsound(write_address)
