from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the pre-trained model
model = load_model('pest_disease_model_vgg16.h5')
class_names = ['stem_borer', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 'sawfly', 'mosquito', 'mites', 'aphids']

# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 255.0
    x = tf.reshape(x, (1, 224, 224, 3))
    predictions = model.predict(x)
    class_index = tf.argmax(predictions, axis=1)[0]
    class_name = class_names[class_index]
    return class_name, img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            class_name, img = predict_image(filepath)

            # Convert the image to display it on the result page
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return render_template('index.html', class_name=class_name, image=img_b64)

    return render_template('index.html', class_name=None, image=None)

if __name__ == '__main__':
    app.run(debug=True)


'''import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('pest_disease_model_vgg19.h5')
class_names=['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']
# Make predictions on a test image 
test_image = "jpg_12.jpg" # path to a test image
img = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = x / 255.0
x = tf.reshape(x, (1, 224, 224, 3))
predictions = model.predict(x)
class_index = tf.argmax(predictions, axis=1)[0]
class_name = class_names[class_index]

# Display the image and predicted class name
plt.imshow(img)
plt.title(class_name)
plt.show()'''