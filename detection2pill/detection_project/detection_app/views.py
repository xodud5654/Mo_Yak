from django.shortcuts import render
import pandas as pd
import json
# from .forms import UploadImageForm
# import base64
# # 객체 감지 로직을 import
# # 예: from detection_module import detect_objects
# import cv2
# import matlab.engine
# import numpy as np
# eng = matlab.engine.start_matlab()
# def detect_objects(request):
#     if request.method == 'POST':
#         form = UploadImageForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#
#             # 이미지 파일 경로
#             image_path = form.instance.image.path
#             print(image_path)
#
#             img = cv2.imread(image_path)
#             img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img,(416,416))
#
#             result = eng.detector(img)
#             print(result)
#             for i in list(result):
#                 print(i)
#                 X,Y,W,H = int(i[0]), int(i[1]), int(i[2]), int(i[3])
#                 cv2.rectangle(img,(X,Y),(X+W,Y+H),(0,255,0),2)
#                 print("done")
#             cv2.imwrite("test.png",img)
#             _, buffer = cv2.imencode('.png', img)
#             img_base64 = base64.b64encode(buffer).decode('utf-8')
#
#
#             # 딥러닝 모델을 사용하여 객체 감지 수행
#             # 결과를 얻고 처리하는 로직
#             # 예: results = detect_objects(image_path)
#
#             # 결과를 템플릿에 전달
#             context = {
#                 'results': list(result),
#                 'img_base64': img_base64,
#
#             }
#             return render(request, 'detection_app/detection_results.html', context)
#     else:
#         form = UploadImageForm()
#     return render(request, 'detection_app/upload_image.html', {'form': form})

# detection_app/views.py

from django.http import JsonResponse
from .forms import UploadImageForm
import base64
# 객체 감지 로직을 import
# 예: from detection_module import detect_objects

from rest_framework.decorators import api_view

import cv2
import matlab.engine
import numpy as np


def detection(image_path):
    print("detecting")
    bbox_img = []
    img = cv2.imread(image_path)
    img = cv2.resize(img, (416, 416))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    eng = matlab.engine.start_matlab()
    result = eng.detector(img_bgr)
    eng.exit()
    print(result)
    detected_img = img
    for i in list(result):

        X, Y, W, H = int(i[0]), int(i[1]), int(i[2]), int(i[3])
        cv2.rectangle(detected_img, (X, Y), (X + W, Y + H), (0, 0, 255), 3)
        if X-20>1 and Y-20>1 and Y+H+40<img.shape[1] and X+W+40<img.shape[0]:
            cropped_img = img_bgr[Y - 20: Y + H + 40, X - 20: X + W + 40]
        else:
            cropped_img = img_bgr[Y: Y + H, X: X + W]

        print(cropped_img.shape)
        bbox_img.append(cropped_img)

    cv2.imwrite("test.png", detected_img)
    print("detection done")
    return detected_img, bbox_img
def classify(bbox_img):
    print("classifiering")
    labels = []
    df = pd.read_csv("labeling.csv")
    df.columns = ["idx", "K", "L"]
    eng = matlab.engine.start_matlab()
    for img in bbox_img:
        img = cv2.resize(img, [224, 224])
        label = eng.classification(img)
        labels.append(str(df[df.K == label].L.values[0]))
    eng.exit()
    print("classify done")
    return labels


from bs4 import BeautifulSoup
import requests
import unicodedata
import time


def crawling(labels):
    print("crawling")
    ALL_text = []
    label = []
    for pill in labels:
        print(pill)
        url = "https://nedrug.mfds.go.kr/searchEasyDrug/easyDetail?itemSeq=" + str(pill)
        response = requests.get(url)
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

        title_element = soup.select_one('.drug_allmenu .title h1 strong')
        if title_element:
            titles = soup.findAll('h3', {'class': 'cont_title2'})
            info_boxes = soup.findAll('div', {'class': 'info_box'})
            text_filed = ''
            for a, b in zip(titles, info_boxes):
                title = a.text.strip()
                title = unicodedata.normalize("NFKD", title)
                info = b.text.strip()
                info = unicodedata.normalize("NFKD", info)
                text_filed += "Q. " + title
                text_filed += "\n"
                text_filed += "A. " + info
                text_filed += "\n\n"
            label.append(title_element.text.strip().split("(")[0])
            ALL_text.append(text_filed)
        else:
            label.append("No result")
            ALL_text.append("No_result")
    print("crawling done")
    print(ALL_text, label)
    return ALL_text, label

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def detect_objects(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # 이미지 파일 경로
            image_path = form.instance.image.path

            detected_img, bbox_img = detection(image_path)
            labels = classify(bbox_img)
            print(labels)
            ALL_text, label = crawling(labels)
            # 처리된 결과 이미지를 Base64로 인코딩
            img_base64 = encode_image_to_base64(detected_img)
            # label,ALL_text=[],[]
            # crawling(labels)

            # 결과를 템플릿에 전달
            context = {
                'labels': json.dumps(label),
                'img_base64': img_base64,
                'ALL_text': json.dumps(ALL_text),
            }
            print("return")
            return render(request, 'detection_app/detection_results.html', context)
    else:
        form = UploadImageForm()
        print("home")
    return render(request, 'detection_app/upload_image.html', {'form': form})
