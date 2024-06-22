from flask import Flask, render_template, request
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)
CORS(app)

def annotationTEXT(captionTEXT):
    textARR = captionTEXT.split('"start"')
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
    captionTEXT = get_youtube_captions(videoID)
    annotationTEXT = annotate_captions(captionTEXT)


    return {"is_video": True, "message": captionTEXT+annotationTEXT} #you have to return json here as explained in the js file


if __name__ == "__main__":
    app.run(debug=True)