from flask import Flask, render_template, request
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
import json
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import sentence_transformers

app = Flask(__name__)
CORS(app)

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

#depickling
corp_embedding = pickle.load(open("embeddings.pkl", "rb"))
corpus = pickle.load(open("corpus.pkl", "rb"))

#global static variables
TOKEN_SIZE = 2
THRESHOLD = 0.65*1000

def preprocess(sentences):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return inputs['input_ids'], inputs['attention_mask']

def get_embeddings(input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        # Take the mean of the hidden states of the tokens (excluding special tokens like [CLS], [SEP])
        embeddings = torch.mean(hidden_states, dim=1)
    return embeddings


def compute_similarity(embeddings1, embeddings2):
    return cosine_similarity(embeddings1, embeddings2)


def process_captions(chunks):
    output = []
    for element in chunks:
        #get the embedding of element[1] ie the text component
        input_id, attention_mask = preprocess(element[1])
        sent_embedding = get_embeddings(input_id, attention_mask)

        #get the top k from embedding space
        results = sentence_transformers.util.semantic_search(sent_embedding, corp_embedding, top_k = 3)
        if not results:
            output.append(element)
        else:
            temp = []
            for r in results[0]:
                temp.append(corpus[r['corpus_id']] + " " + str(r['score'])) 
            output.append([element[0], element[1], temp])

    return output


def chunk_captions(captionARR):
    textARR = []
    temp = ""
    idx = 0
    size = len(captionARR)
    timestamp = 0 #a number
    while idx < size-1:
        idx+=1
        e = captionARR[idx]
        if idx%TOKEN_SIZE == 0 or idx == size-1:
            textARR.append([timestamp, temp]);
            temp = ""
            timestamp = float(e['start'])
        temp += " " + e['text']
    return textARR

def get_youtube_captions(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return None 


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/captions", methods=["GET"])
def captions():
    videoURL = request.args['id']
    if videoURL.find('=') == -1:
        return {"is_video": False, "message": 'Not watching a YouTube video'}
    videoID = videoURL.split('=')[1]
    captionARR = get_youtube_captions(videoID)
    chunkARR = chunk_captions(captionARR)
    response = process_captions(chunkARR)


    return {"is_video": True, "message": json.dumps(response)} #you have to return json here as explained in the js file


if __name__ == "__main__":
    app.run(debug=True)