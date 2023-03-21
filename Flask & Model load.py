import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed file types
app.config['UPLOAD_FOLDER'] =r"uploads"
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

# Check if uploaded file is an allowed file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
# Render the upload form
@app.route('/')
def upload_form():
    return render_template('HTML and CSS webpage configuration.html')

# Handle the image upload and apply ML code
@app.route('/', methods=['POST'])
def upload_image():
    # Check if file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    # Check if file is an allowed file type
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    # Save file to upload folder
    filename = secure_filename(file.filename)

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Apply ML code to uploaded image here
    # ...
    upload_folder = app.config['UPLOAD_FOLDER']
    filepath = os.path.join(upload_folder, filename)

   # Imported necessary libraries
    from tensorflow import keras
    import tensorflow as tf
    import cv2
    import numpy as np
    

    # Loading the model
    model = keras.models.load_model('/content/final_model.h5')
    model.compile()

    # Reading image using imread method
    img = cv2.imread(filepath)
    img_str = cv2.imencode('.jpg', img)[1].tostring()
    

    # Converting image to greyscale and resizing 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray_img = np.expand_dims(gray_img, axis=-1)      #adding dimension because tf.image.resize() required 3D input grey scale has 2D
    resize = tf.image.resize(gray_img, (128, 128))
    # plt.imshow(resize.numpy().astype(int), cmap='gray') 
    # plt.show()

    yhat = model.predict(np.expand_dims(resize/255, 0))
    yhat=1-yhat   #inference model got train on positive class as unconfident 
    print(yhat)


    if yhat >= 0.55:
        message = 'Predicted Level of Confidence : High'
    elif yhat <= 0.30:
        message = 'Predicted Level of Confidence : Low'
    else:
        message = 'Predicted Level of Confidence : Neutral'
    

    return render_template('HTML and CSS webpage configuration.html', message=message, image=img_str)

    return 'File uploaded and ML code applied successfully!'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)