import os
import base64
from flask import Flask, render_template, request, jsonify
import replicate

app = Flask(__name__)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_video():
    if not REPLICATE_API_TOKEN:
        return jsonify({'error': 'Replicate API key not set on Render!'}), 500
        
    data = request.json
    image_data = data.get('image_data') # Base64 string from gallery
    prompt = data.get('prompt', 'Animate this image smoothly in high quality')

    if not image_data:
        return jsonify({'error': 'Please select an image!'}), 400

    try:
        # Running Minimax Video-01 Model (Supports Image + Text Prompt)
        output = replicate.run(
            "minimax/video-01",
            input={
                "first_frame_image": image_data,
                "prompt": prompt
            }
        )
        
        # If output is FileOutput object or URL
        video_url = output.url if hasattr(output, 'url') else str(output)
        return jsonify({'video_url': video_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
