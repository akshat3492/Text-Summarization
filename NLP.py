#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install datasets


# In[2]:


#pip install transformers


# In[3]:


#pip install evaluate


# In[4]:


#pip install sentencepiece


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
#nltk.download("punkt")
import re
import transformers
import sentencepiece
import datasets
from datasets import load_dataset, DatasetDict
from datasets import Dataset
import evaluate


# In[2]:


cnn_train_dataset = load_dataset("cnn_dailymail", '3.0.0', split='train[:2500]')
cnn_validation_dataset = load_dataset("cnn_dailymail", '3.0.0', split='validation[:200]')
cnn_test_dataset = load_dataset("cnn_dailymail", '3.0.0', split='test[:50]')

# Create a DatasetDict manually
combined_dataset = DatasetDict({
    "train": cnn_train_dataset,
    "validation": cnn_validation_dataset,
    "test": cnn_test_dataset
})


# In[3]:


combined_dataset


# # Train

# In[4]:


train = combined_dataset['train'].to_pandas()
#train = train.iloc[:10000]


# In[5]:


#Check for duplicates
train[train.duplicated()]


# In[6]:


#Check for Null Values
train[train.isnull().any(axis=1)]


# In[7]:


#make all words lower to ensure consistency
train = train[['article', 'highlights']]
train = train.apply(lambda x: x.astype(str).str.lower())


# In[8]:


#Checking word distribution
def count_words(text):
    return len(text.split())


# In[9]:


for column in train.columns:
    train[column + '_word_count'] = train[column].apply(count_words)

# Create separate histograms vertically
fig, axes = plt.subplots(nrows=len(train.columns[2:]), ncols=1, figsize=(8, 6 * len(train.columns[2:])))
plt.subplots_adjust(hspace=0.5)

for idx, column in enumerate(train.columns[2:]):
    ax = axes[idx]
    ax.hist(train[column], bins=20, alpha=0.5)
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of Word Counts in {column}')

plt.show()


# In[10]:


#remove html tags & URLs

def clean_text(text):
    # Remove HTML tags
    text_no_tags = re.sub(r'<[^>]*>', '', text)

    # Remove URLs
    text_no_urls = re.sub(r'http\S+', '', text_no_tags)

    return text_no_urls

train['article'] = train['article'].apply(clean_text)
train['highlights'] = train['highlights'].apply(clean_text)


#remove specfic punctuations
def clean_punctuations(text):
    # Define punctuation marks to remove
    punctuation_to_remove = ['-- ']

    # Remove specified punctuation
    for punct in punctuation_to_remove:
        text = text.replace(punct, '')

    # Remove text between parentheses
    text = re.sub(r'\([^)]*\) ', '', text)

    return text

train['article'] = train['article'].apply(clean_punctuations)
train['highlights'] = train['highlights'].apply(clean_punctuations)


# In[11]:


train = Dataset.from_pandas(train)


# In[12]:


combined_dataset['train'] = train


# In[13]:


combined_dataset


# # Validation

# In[14]:


test = combined_dataset['validation'].to_pandas()


# In[15]:


#Check for duplicates
test[test.duplicated()]


# In[16]:


#Check for Null Values
test[test.isnull().any(axis=1)]


# In[17]:


#make all words lower to ensure consistency
test = test[['article', 'highlights']]
test = test.apply(lambda x: x.astype(str).str.lower())


# In[18]:


#remove html tags & URLs

def clean_text(text):
    # Remove HTML tags
    text_no_tags = re.sub(r'<[^>]*>', '', text)

    # Remove URLs
    text_no_urls = re.sub(r'http\S+', '', text_no_tags)

    return text_no_urls

test['article'] = test['article'].apply(clean_text)
test['highlights'] = test['highlights'].apply(clean_text)

#remove specfic punctuations
def clean_punctuations(text):
    # Define punctuation marks to remove
    punctuation_to_remove = ['-- ']

    # Remove specified punctuation
    for punct in punctuation_to_remove:
        text = text.replace(punct, '')

    # Remove text between parentheses
    text = re.sub(r'\([^)]*\) ', '', text)

    return text

test['article'] = test['article'].apply(clean_punctuations)
test['highlights'] = test['highlights'].apply(clean_punctuations)


# In[19]:


test = Dataset.from_pandas(test)
combined_dataset['validation'] = test


# In[20]:


combined_dataset


# ## Tokenization

# In[21]:


pip install sentencepiece


# In[22]:


from transformers import AutoTokenizer

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[23]:


max_input_length = 800
max_target_length = 50


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["article"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["highlights"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[24]:


tokenized_datasets = combined_dataset.map(preprocess_function, batched=True)


# In[25]:


tokenized_datasets


# # Baseline Model

# In[26]:


import nltk

nltk.download("punkt")


# In[27]:


from nltk.tokenize import sent_tokenize


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


# In[28]:


def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["article"]]
    return metric.compute(predictions=summaries, references=dataset["highlights"])


# In[29]:


get_ipython().system('pip install rouge_score')


# In[30]:


import evaluate
rouge_score = evaluate.load("rouge")


# In[31]:


import pandas as pd

score = evaluate_baseline(combined_dataset["validation"], rouge_score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


# In[32]:


score


# # Fine Tune mT5

# In[33]:


from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# In[34]:


from huggingface_hub import notebook_login

notebook_login()


# In[39]:


#pip install accelerate -U


# In[40]:


#pip install transformers[torch]


# In[35]:


from transformers import Seq2SeqTrainingArguments

batch_size = 8
num_train_epochs = 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"mT5",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)


# In[36]:


import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    print(result)
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


# In[37]:


from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[38]:


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# In[39]:


trainer


# In[40]:


trainer.train()


# In[41]:


trainer.evaluate()


# In[42]:


trainer.push_to_hub(commit_message="Training complete", tags="summarization")


# In[43]:


from transformers import pipeline

hub_model_id = "akshat3492/mT5"
summarizer = pipeline("summarization", model=hub_model_id)


# In[47]:


def print_summary(idx):
    review = combined_dataset["test"][idx]["article"]
    title = combined_dataset["test"][idx]["highlights"]
    summary = summarizer(combined_dataset["test"][idx]["article"])[0]["summary_text"]
    print(f"'>>> Review: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> Summary: {summary}'")


# In[49]:


print_summary(8)


# In[50]:


#from transformers import AutoModelForTextSummarization

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("akshat3492/mT5")


# In[51]:


model = AutoModelForSeq2SeqLM.from_pretrained("akshat3492/mT5")


# In[52]:


input_text = '(CNN)A Duke student has admitted to hanging a noose made of rope from a tree near a student union, university officials said Thursday. The prestigious private school didn\'t identify the student, citing federal privacy laws. In a news release, it said the student was no longer on campus and will face student conduct review. The student was identified during an investigation by campus police and the office of student affairs and admitted to placing the noose on the tree early Wednesday, the university said. Officials are still trying to determine if other people were involved. Criminal investigations into the incident are ongoing as well. Students and faculty members marched Wednesday afternoon chanting "We are not afraid. We stand together,"  after pictures of the noose were passed around on social media. At a forum held on the steps of Duke Chapel, close to where the noose was discovered at 2 a.m., hundreds of people gathered. "You came here for the reason that you want to say with me, \'This is no Duke we will accept. This is no Duke we want. This is not the Duke we\'re here to experience. And this is not the Duke we\'re here to create,\' " Duke President Richard Brodhead told the crowd. The incident is one of several recent racist events to affect college students. Last month a fraternity at the University of Oklahoma had its charter removed after a video surfaced showing members using the N-word and referring to lynching in a chant. Two students were expelled. In February, a noose was hung around the neck of a statue of a famous civil rights figure at the University of Mississippi. A statement issued by Duke said there was a previous report of hate speech directed at students on campus. In the news release, the vice president for student affairs called the noose incident a "cowardly act." "To whomever committed this hateful and stupid act, I just want to say that if your intent was to create fear, it will have the opposite effect," Larry Moneta said Wednesday. Duke University is a private college with about 15,000 students in Durham, North Carolina. CNN\'s Dave Alsup contributed to this report.'
inputs = tokenizer(input_text, return_tensors="pt", max_length=800, truncation=True, padding=True)


# In[53]:


summary = model.generate(**inputs)


# In[54]:


decoded_summary = tokenizer.decode(summary[0], skip_special_tokens=True)


# In[56]:


decoded_summary


# In[20]:


import pickle


# In[21]:


pickle.dump(model, open('model.pkl', 'wb'))


# In[ ]:




