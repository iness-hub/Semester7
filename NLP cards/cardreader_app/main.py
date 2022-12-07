from flask import Flask, request
from flask import render_template
import settings
import utils
import numpy as np
import cv2
import prediction as pred

app = Flask(__name__)
app.secret_key = 'invoices_reader_app'

# docscan = utils.DocumentScan()


@app.route('/',methods=['GET','POST'])
# @app.route('/')
def readInvoice():
    
    if request.method == 'POST':
        file = request.files['image_name']
        upload_image_path = utils.save_upload_image(file)
        print('Image saved in = ',upload_image_path)  
        
        return render_template('reader.html', fileupload=True)
    
    return render_template('reader.html')

@app.route('/prediction')
def prediction():
    # load the wrap image
    wrap_image_filepath = settings.join_path(settings.MEDIA_DIR,'upload.jpg') 

    image = cv2.imread(wrap_image_filepath)
    image_bb  = pred.getPredictions(image)
    
    bb_filename = settings.join_path(settings.MEDIA_DIR,'bounding_box.jpg') 
    cv2.imwrite(bb_filename,image_bb)
    
    return render_template('predictions.html')


@app.route('/about')
def about():
    return render_template('about.html')



if __name__ == "__main__":
    app.run(debug=True)