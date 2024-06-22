from flask import Flask, render_template, request
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)
CORS(app)

#global static variables
TOKEN_SIZE = 5

def annotate_captions(captionARR):
    textARR = []
    temp = []
    idx = 0
    size = len(captionARR)
    while idx < size:
        idx+=1
        if idx%TOKEN_SIZE == 0 || idx == size-1:
            temp.append(idx)
            textARR.append(temp)
            temp = []
        e = captionARR.get(idx)
        temp.append(e)
    return str(textARR)

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


    return {"is_video": True, "message": annotationTEXT + "*****" + captionARR} #you have to return json here as explained in the js file


if __name__ == "__main__":
    app.run(debug=True)