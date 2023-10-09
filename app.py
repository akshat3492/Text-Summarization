#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__, template_folder='templates')

model = T5ForConditionalGeneration.from_pretrained("akshat3492/mT5")
tokenizer = T5Tokenizer.from_pretrained("akshat3492/mT5")

def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=800, truncation=True, padding=True)
    summary_ids = model.generate(inputs, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=50, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/summarize', methods=['GET','POST'])
def summarization():
    text = request.form['text']
    summary = summarize_text(text)
    return render_template('index1.html', original_text=text, summary=summary)#text=text, summary=summary)


# In[2]:


if __name__ == '__main__':
    app.run()


# In[ ]:




