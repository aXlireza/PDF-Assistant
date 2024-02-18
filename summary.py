from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

# %% [markdown]
# ** SUMMERIZATION models

# %%
# facebook/bart-large-cnn
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def facebook_bart_large_cnn(text):
    summary = summarizer(text, max_length=20 if len(text.split())<=50 else 70, min_length=10, do_sample=False)
    return summary[0]['summary_text']

# %%
# # google/bigbird-pegasus-large-bigpatent
# summarytokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-bigpatent")
# summarymodel = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-bigpatent", attention_type="original_full")
# def google_bigbird_pegasus_large_bigpatent(text):
#     inputs = summarytokenizer(text, return_tensors='pt')
#     summaryprediction = summarymodel.generate(**inputs)
#     thesummary = summarytokenizer.batch_decode(summaryprediction)
#     return thesummary[0].replace('<s>', '').replace('</s>', '').strip()

# %%
def summerize(text):
    # print(len(text.split()))
    # print(textwrap.fill(text, 80))

    return facebook_bart_large_cnn(text)
    # return google_bigbird_pegasus_large_bigpatent(text)
