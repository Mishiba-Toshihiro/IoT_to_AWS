import sys
import argparse
import numpy as np
import time # 追加
import random # 追加
from yolo import YOLO, detect_video
from PIL import Image
"""
def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()
"""
def detect_img(yolo):
    input('準備完了。[enter]を押して開始してください')
    try :
        while True:
            #テストなのでランダムに画像を取得し解析した画像を保存。約10秒ごとに保存された解析画像が切り替わるか確認
            num = random.randint(1,10)
            if num < 4:
                img = 'face.jpg'
            elif num < 7:
                img = 'face2.jpg'
            else:
                img = 'face3.jpeg'

            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                # r_image.show()

                # 画像の書き出し、10秒ごとに書き出す
                import cv2 
                cv2.imwrite("output/output_image.jpg", np.asarray(r_image)[..., ::-1])
                time.sleep(90)

        yolo.close_session()

    except KeyboardInterrupt:
        # cotrol + c で停止できるように KeyboardInterruptで例外をキャッチ
        print("停止しました")


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    detect_img(YOLO(**vars(FLAGS))) 
"""
    if FLAGS.image:
        Image detection mode, disregard any remaining command line arguments
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
"""