# %%
import os
import re
# from pynput import keyboard
from plyer import notification
from num2words import num2words
import summary
import voice
# this is 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.system("export CUDA_VISIBLE_DEVICES=\"\"")


# %%
def show_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        timeout=10
    )

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
        thesummary = originaltext if len(originaltext.split())<=10 else summary.summerize(originaltext)
        thesummarysplitted = [substr for substr in re.split(r"[.!?;:]", thesummary) if substr]
        print(thesummarysplitted)
        for tmptext in thesummarysplitted: voice.readoutload(tmptext)
    elif mode == 'tts':
        thesummarysplitted = [substr for substr in re.split(r"[.!?;:]", originaltext) if substr]
        print(thesummarysplitted)
        for tmptext in thesummarysplitted: voice.readoutload(tmptext)
