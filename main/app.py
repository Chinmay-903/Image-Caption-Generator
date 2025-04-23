import os
import requests
from flask import Flask, render_template, request
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

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

@app.route('/', methods=['GET', 'POST'])
def index():
    captions = None
    image_path = None

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

    return render_template('index.html', captions=captions, image_path=image_path)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
