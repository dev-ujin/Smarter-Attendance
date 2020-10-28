from flask import Flask, render_template, request, send_from_directory
import os
from findOverlappedArea.Utils import loadImagesInFolder, createFolder
from findOverlappedArea.OverlappedAreaFinder import findOverlappedAreaInAllImages
import cv2
from flask_bootstrap import Bootstrap
from faceDetection.mtcnn import MTCNN
from faceDetection.FaceDetector import facedetector

app = Flask(__name__)
Bootstrap(app)
globalcount = 0

@app.route('/')
def home():
    return render_template('about.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload() :
    return render_template('upload.html')

@app.route('/uploader', methods=['POST', 'GET'])
def uploader() :
    path = './image/input_image/'
    for f in request.files.getlist('file[]') :
        f.save(path + f.filename)
    image_dir = os.listdir('./image/input_image/')
    return render_template("uploadresult.html", image_dir = image_dir)

@app.route('/uploader/<filename>')
def send_image(filename) :
    return send_from_directory('./image/input_image', filename)

def findoverlappedarea() :
    # set input/output path
    inputImagePath = './image/input_image/'
    outputImagePath = './image/overlapped_image/'

    # load images
    print('Image loading...', '(Image file path : ' + inputImagePath + ')')
    images = loadImagesInFolder(inputImagePath)
    print('Complete image load!')
    print()

    # Every image stitch and get result
    print('Find overlapped area in all images...')
    overlappedAreaImages = findOverlappedAreaInAllImages(images)
    print()

    if overlappedAreaImages is not None:
        # create output folder
        createFolder(outputImagePath)

        # extract images(overlapped area), images(erased overlapped area),
        onlyOverlappedAreaImages, overlappedAreaDrawedImages = overlappedAreaImages

        # result files save
        print('File saving...')
        for i, e in enumerate(onlyOverlappedAreaImages):
            cv2.imwrite(outputImagePath + 'BF_ORB_onlyOverlappedArea_' + str(i + 1) + '_' + str(i + 2) + '.png', e)
        cv2.destroyAllWindows()
    return

def facedetection() :
    detector = MTCNN()

    #global variable for counting
    global globalcount

    # set input/output path
    inputImagePath_originalImage = './image/input_image/'
    inputImagePath_overlappedImage = './image/overlapped_image/'
    outputImage = './image/output_image/'

    # for original images
    images = os.listdir(inputImagePath_originalImage)
    i=0
    for image in images :
        outputimage, count = facedetector(inputImagePath_originalImage, image)
        cv2.imwrite(outputImage + 'result' + str(i)+ '.jpg', outputimage)
        print('complete save result' + str(i) + '.jpg')
        setcount(getcount() + count)
        i+=1

    images = os.listdir(inputImagePath_overlappedImage)
    #for overlapped images
    for image in images:
        outputimage, count = facedetector(inputImagePath_overlappedImage, image)
        cv2.imwrite(outputImage + 'result' + str(i) + '.jpg', outputimage)
        print('complete save result' + str(i) + '.jpg')
        setcount(getcount() - count)
        i+=1

def setcount(c):
    global globalcount
    globalcount=c

def getcount():
    global globalcount
    return globalcount

@app.route('/result', methods=['POST', 'GET'])
def result():
    global globalcount
    findoverlappedarea()
    facedetection()
    image_dir_input = os.listdir('./image/input_image')
    length = len(image_dir_input)
    image_dir_output = os.listdir('./image/output_image/')
    return render_template("result.html", image_dir_input = image_dir_input, image_dir_output=image_dir_output, totalcount=globalcount, length=length)

@app.route('/result/<filename>')
def send_image_output(filename) :
    return send_from_directory('./image/output_image', filename)

if __name__ == '__main__':
    app.run(debug=True)
