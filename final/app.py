from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from PIL import Image
from io import BytesIO
import json

app = Flask(__name__)

# Load the pre-trained model
model = load_model('mnist_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request)
        # Get the image file from the API call
        file = request.files['file']
        print("Before image load")

         # Read the content of the file
        file_content = file.read()
        img = Image.open(BytesIO(file_content)).convert('L')  # 'L' = grayscale

        if img.size != (28,28):
            return jsonify({'error': f'Image dimensions should be (28,28)'})
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0 # normalize data
        
        print("past image load")
        # Make predictions
        predictions = model.predict(img_array)

        # Interpret the results
        predicted_digit = np.argmax(predictions)
        print(predictions)
        
        # Return the result as JSON
        result = {'predicted_digit': int(predicted_digit), 'confidence': predictions.tolist()[0][int(predicted_digit)], 'all': predictions.tolist()}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
