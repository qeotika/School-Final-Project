
from flask import Flask, request, jsonify
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True #Fixed backend error
import sys
from PIL import Image
sys.path.append('C:\Programming PYTHON\MarkRcnn37Dig\Mask-RCNN-TF2')
from main import proccess_images_data
#Routes to "site" ..../home or .../testme ect
app = Flask(__name__)
@app.route('/')
def home():
    return 'Connected to server. Waiting for request'
#For connection test
@app.route('/test', methods=['POST'])
def testme():
    return 'hello'

@app.route('/process_images', methods=['POST'])
def process_images():
    for key,value in request.files.items():
        print(f"Key: {key}")
        print(f"Value: {value}")
    print('Request Method:' , request.method)
    print('Request URL:' , request.url)
    print('Request Headers:' , request.headers)
    print('Request Body: ' , request.get_data(as_text=True))

    
    if 'imageSide' in request.files or 'imageTop' in request.files:
        print("Inside the if statement")
        image1 = request.files['imageSide']
        print("First image uploaded")
        print(str(image1))
        if image1 is None:
            print("Empty image1")
        image2 = request.files['imageTop']
        print("Second image uploaded")
        print(image2)
        image1 = Image.open(image1)
        image2 = Image.open(image2)
        np_image = image1.convert('RGB')
        np_image2 = image2.convert('RGB')
        image1_path = "C:/Programming PYTHON/MarkRcnn37Dig/Mask-RCNN-TF2/frontflask/toppic.jpg"
        image2_path = "C:/Programming PYTHON/MarkRcnn37Dig/Mask-RCNN-TF2/frontflask/sidepic.jpg"
        np_image.save(image1_path)
        np_image2.save(image2_path)
        
        
        
        
        
        # Call the main() function from the image_processing module
        print("here we go")
        processed_results_json = proccess_images_data(image1_path, image2_path)

        print("RESPONSE TO CLIENT:")
        # Set the response headers to indicate JSON content type
        response = app.response_class(
                response=processed_results_json,
                status=200,
                mimetype='application/json'
            )
        print("Images uploaded successfully")
        print(response)
        return response
      
      
        
    else:
        print("Bad request try again")
        response = {'error' : 'Missing images in the request'}
        return jsonify(response), 400

if __name__ == '__main__':
    app.run(port=7000,debug=False, threaded=False)