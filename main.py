from flask import Flask, render_template, request
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
import json
import pickle4 as pickle
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import heap

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')

#depickling
corp_embedding = pickle.load(open("embeddings.pkl", "rb"))
corpus = pickle.load(open("corpus.pkl", "rb"))

app = Flask(__name__)
CORS(app)

#global static variables
TOKEN_SIZE = 5 #cuts into about 10 sec pieces

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

def semantic_search_jr(sentence_embed, corpus_embed, top_k = 10): #uses cosine similarity
    heap = []
    min_sim = 0
    similarity_scores = compute_similarity(sentence_embed, corpus_embed)[0]
    for x in range(len(similarity_scores)):
        score = similarity_scores[x]
        if  score > min_sim:
            text = corpus[x]
            heappush(heap, (score, text))
            if len(heap) > top_k:
                heappop(heap)
    return heapsort(heap)


def compute_similarity(embeddings1, embeddings2):
    return cosine_similarity(embeddings1, embeddings2)


def process_captions(chunks):
    for element in chunks:
        #get the embedding of element[1] ie the text component
        input_id, attention_mask = preprocess(element[1], tokenizer)
        sent_embedding = get_embeddings(input_id, attention_mask)

        #get the top k from embedding space
        results = semantic_search_jr(sent_embedding, corp_embedding)[0]
        for r in results:
          print(corpus[r['corpus_id']] +" score: "+ str(r['score']) + "\n")
        
        element.append(results)


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


    return {"is_video": True, "message": json.dumps(chunkARR)} #you have to return json here as explained in the js file


if __name__ == "__main__":
    app.run(debug=True)