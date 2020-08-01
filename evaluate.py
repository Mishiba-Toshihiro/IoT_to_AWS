# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import time # 追加
import random # 追加
import picamera#追加
from time import sleep#追加
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import json

import paho.mqtt.client as mqtt#ここからaws用に追加
import ssl
port = 8883
cert = "./cert/.pem.crt"#モノの証明書
key = "./cert/.pem.key"#プライベートキー
rootCA = "./cert/.pem"#Amazonルート証明書
endpoint = ".com"
clientID = "RasPi02"
topic = "count/pub"#人数（混雑率）をRaspberryPi to AWS
terminalID = "SK-RPi001"

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        print("\n")
        print("解析中.....")
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        
        person_num = 0#初期値追加
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))
            
            if predicted_class == "person":#人の場合カウントするように追加
                person_num+=1
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print('--------結果--------'+' (解析時間'+ ' {:.1f} '.format(end - start)+ '秒)')
        print("人数：" + str(person_num))  #画像に映る人数を表示する行を追加
        #S = 10*10    #カメラ測定範囲の面積をあらかじめ設定。デフォルトでは10m×10mとしている。
        konzatu = person_num/self.S    #混雑率を人数/面積[人/m2]で定義
        kijyunnnisnnsuu = 0.08 * self.S
        #kijyunnnisnnsuu = round(kijyunnnisnnsuu,0)
        print("混雑率：" + str(konzatu) + " [人/m2]")
        if konzatu>=0.08:
            print("対人距離2m確保するための基準値0.08[人/m2]を超えています (ﾉﾟ□ﾟ)ﾉ")
            #print("基準値以下にするには、"+ str(kijyunnnisnnsuu)+"[人]以下である必要があります。"))
        else:
            print("対人距離2m確保するための基準値0.08[人/m2]を下回っているため問題ありません ^_^")
        date = datetime.now()
        date_str = date.strftime('%Y%m%d %H:%M:%S')
        self.d_new = {"Date": date_str, "Person_num": person_num, "Congestion": konzatu}
        #{"日時": date_str, "人数": person_num, "混雑率[人/m2]": konzatu} SQLでうまく読み込めなかったので上に変えた
        with open("/home/pi/keras-yolo3/out_json/out.json",'w')as f:
            json.dump(self.d_new,f,ensure_ascii=False)
        return image

    def close_session(self):
        self.sess.close()



def detect_img(yolo):
    print("\n")
    print("RaspberryPi to AWS Start!!")
    print("\n")
    client = mqtt.Client(clientID, protocol=mqtt.MQTTv311)#初期化
    client.tls_set(ca_certs=rootCA,
                   certfile=cert,
                   keyfile=key,
                   cert_reqs=ssl.CERT_REQUIRED,
                   tls_version=ssl.PROTOCOL_TLSv1_2,
                   ciphers=None)
    client.connect(endpoint,port=port,keepalive=60)#AWS IoTに接続
    client.loop_start()#ループスタート
    yolo.S=float(input('撮影場所の広さ[m2]を入力し、[enter]を押してください。'))
    print("\n")
    input('準備完了。[enter]を押して開始してください。')
    print("\n")
    try :
        while True:
            #テストなのでランダムに画像を取得し解析した画像を保存。約10秒ごとに保存された解析画像が切り替わるか確認
            #num = random.randint(1,10)
            #if num < 4:
               # img = 'face.jpg'
            #elif num < 7:
             #   img = 'face2.jpg'
            #else:
              #  img = 'face3.jpeg'
            with picamera.PiCamera() as camera:
                camera.resolution = (416,416)
                camera.start_preview()
                sleep(2)
                camera.capture('/home/pi/keras-yolo3/data/data.jpg')
            try:
                image = Image.open('/home/pi/keras-yolo3/data/data.jpg')
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                client.publish(topic,json.dumps(yolo.d_new))#AWS IoTに送信（publish）
                # r_image.show()

                # 画像の書き出し、90秒ごとに書き出す
                import cv2 
                cv2.imwrite("output/output_image.jpg", np.asarray(r_image)[..., ::-1])
                time.sleep(90)

        yolo.close_session()
    except KeyboardInterrupt:
        # cotrol + c で停止できるように KeyboardInterruptで例外をキャッチ
        print("停止しました")
        client.disconnect()#停止時にdiconnect


if __name__ == '__main__':
    detect_img(YOLO())