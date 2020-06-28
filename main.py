from flask import *
import keras
import numpy as np
from keras.preprocessing import image
app = Flask('app')
@app.route('/')
def scan():
  return render_template("scan.html")
####################################
@app.route('/result', methods = ["GET",'POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)
        
        classifier = keras.models.load_model('model progress')

        test_image = image.load_img('user x-ray pictures/' + f.filename, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)

        if result[0][0] > 0.7:
          return render_template("corona.html")
        elif result[0][0] > 0.3:
          return render_template("maybe.html")
        else: 
          return render_template("nocorona.html")
app.run(host='0.0.0.0', port=8080)
####################################
