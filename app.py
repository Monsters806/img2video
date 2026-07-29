import os
import time
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
        return jsonify({'error': 'API key not configured on server'}), 500
        
    data = request.json
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({'error': 'Image URL required!'}), 400

    try:
        # Standard Stable Video Diffusion model format
        output = replicate.run(
            "stability-ai/stable-video-diffusion",
            input={
                "input_image": image_url,
                "motion_bucket_id": 127,
                "cond_aug": 0.02
            }
        )
        return jsonify({'video_url': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
