from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .predictEmotion import emotionPred
import os
import serial
import time
arduino = serial.Serial(port='COM10', baudrate=115200, timeout=.1)

cwd = os.getcwd()

def home(request):
    return render(request, 'core/Home.html')

def contact(request):
    return render(request, 'core/Contact.html')


def predict(request):
    if request.method == 'POST' and request.FILES['myfile']:

        myfile = request.FILES['myfile']
        print(myfile.size)
        print(myfile.content_type)

        if myfile.content_type.split("/")[1] != "vnd.ms-excel":
            return render(request, 'core/Predict.html', {
                'error_file': "Error : Please Upload a CSV File",
                'uploaded_file_url': ""
            })
        if myfile.size > 23068672:
            return render(request, 'core/Predict.html', {
                'error_file': "Error : File size Exceeded 25 MB",
                'uploaded_file_url': ""
            })

        try:
            fs = FileSystemStorage()
            filename = fs.save("dataFile.csv", myfile)
            print("Filename: ", filename)
            uploaded_file_url = fs.url(filename)
            print("Uploaded file URL: ", uploaded_file_url)
            output_class, output_val = emotionPred(uploaded_file_url)
            # output_val = "0"
            os.remove(cwd + uploaded_file_url)
            # attach_file_name = output_file
            # Open the file as binary mode
            print(output_val, type(output_val))
            arduino.write(bytes(output_val, 'utf-8'))
            time.sleep(0.05)
            # data = arduino.readline()

            return render(request, 'core/Predict.html', {
                'error_file': output_class,
                # 'uploaded_file_url': output_file
            })
        except Exception as e:
            return render(request, 'core/Predict.html', {
                # 'error_file': "Error : Some Error Occured",
                'error_file': str(e),
                'uploaded_file_url': ""
            })
    return render(request, 'core/Predict.html', {
        'uploaded_file_url': ""
    })
