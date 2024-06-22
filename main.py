from flask import Flask, render_template, request
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)
CORS(app)

#global static variables
TOKEN_SIZE = 5

def annotate_captions(captionARR):
    textDICT = {}
    temp = ""
    idx = 0
    size = len(captionARR)
    timestamp = 0 #a number
    while idx < size-1:
        idx+=1
        e = captionARR[idx]
        if idx%TOKEN_SIZE == 0 or idx == size-1:
            textDICT[timestamp] = temp
            temp = ""
            timestamp = float(e['start'])
        temp += e['text']
    return str(textDICT)

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
    annotationTEXT = annotate_captions(captionARR)


    return {"is_video": True, "message": annotationTEXT + "*****" + str(captionARR)} #you have to return json here as explained in the js file


if __name__ == "__main__":
    app.run(debug=True)