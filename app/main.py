from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

app = FastAPI()

# Prompt template
yt_prompt = """
Resuma o video.
Transcript: {transcript}"""

# Model for the request body
class VideoURL(BaseModel):
    url: str

# Function to get transcript from YouTube
def get_transcript(video_id, languages=['pt']):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        # Combine all transcript segments into one string
        combined_transcript = " ".join([segment['text'] for segment in transcript])
        return combined_transcript
    except NotImplemented:
        raise HTTPException(status_code=404, detail="No transcript found for the specified languages.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to summarize video transcript
def summarize_video_ollama(transcript, template=yt_prompt, model="mistral"):
    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format_messages(transcript=transcript)
    ollama = ChatOllama(model=model, temperature=0.1)
    summary = ollama.invoke(formatted_prompt)  # Usando invoke em vez de __call__
    return summary

@app.post("/summarize/")
async def summarize_video(video: VideoURL):
    try:
        # Extract video ID from URL
        video_id = video.url.split("v=")[-1].split("&")[0]
        
        # Get transcript
        transcript = get_transcript(video_id)
        
        # Generate summary using Ollama
        summary = summarize_video_ollama(transcript=transcript)

        return {"summary": summary.content}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
