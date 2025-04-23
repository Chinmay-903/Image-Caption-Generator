import os
import requests
from flask import Flask, render_template, request, send_file
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from gtts import gTTS  # Importing the gTTS library

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and processor once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_captions(image: Image.Image):
    image = image.convert("RGB")

    # Conditional
    conditional_prompt = "a photography of"
    inputs = processor(image, conditional_prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)

    # Unconditional
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    unconditional_caption = processor.decode(out[0], skip_special_tokens=True)

    return conditional_caption, unconditional_caption

# Function to convert text to speech using gTTS
def text_to_speech(text: str, filename: str):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    captions = None
    image_path = None
    audio_file = None  # Variable to store audio file path

    if request.method == 'POST':
        # Handle image URL
        img_url = request.form.get('image_url')
        if img_url:
            response = requests.get(img_url, stream=True)
            image = Image.open(response.raw)
            captions = generate_captions(image)
            image_path = img_url

        # Handle file upload
        if 'image_file' in request.files:
            file = request.files['image_file']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                image = Image.open(filepath)
                captions = generate_captions(image)
                image_path = '/' + filepath

        # Convert the generated captions to speech (if captions exist)
        if captions:
            conditional_caption, unconditional_caption = captions
            combined_caption = f"Conditional caption: {conditional_caption}. Unconditional caption: {unconditional_caption}"
            audio_filename = 'static/audio/caption_audio.mp3'
            text_to_speech(combined_caption, audio_filename)
            audio_file = audio_filename

    return render_template('index.html', captions=captions, image_path=image_path, audio_file=audio_file)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/audio', exist_ok=True)  # Create the 'audio' folder for storing audio files
    app.run(debug=True)
