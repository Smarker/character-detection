from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from PIL import Image

results_dir = os.path.join(os.path.pardir, '..', 'data', 'results')

def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img,image_name,boxes,scale):
    base_name = image_name.split('/')[-1]

    with open(results_dir + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            
            line = ','.join([str(min_x),str(min_y),str(max_x),str(max_y)])+'\r\n'
            f.write(line)
            crop_function(image_name,min_y,min_x,max_y,max_x)

    img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(results_dir, base_name), img)

def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()
    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))


def crop_function(image,left,top,right,bottom):
    image_url=image
    img=Image.open(image_url)
    size=img.size
    box=(left,size[1]-bottom,right,size[1]-top)
    crop_area=img.crop(box)
    indice=0

    x=image_url.split(".")
    s=""
    y=x[0].split("/")
    l=len(y)
    i=0
    for i in range(l):
        if (i==l-1):
            s=s+"results/"+y[i]
        else:
            s=s+y[i]+"/"
    test=True
    while test:
        crop_url=s+"_crop_"+str(indice)+"."+x[1]
        try:
            with open(crop_url): pass
        except IOError:
            test=False
        indice+=1
    crop_area.save(crop_url)


if __name__ == '__main__':
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    cfg_from_file(os.path.join(os.path.dirname(__file__), 'text.yml'))

    sys.path.append(os.path.dirname(__file__))

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.pardir, '..', 'checkpoints'))

        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found!')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    demo_path_pattern = '/mnt/batch/tasks/shared/LS_root/mounts/external/electric/data/demo/'
    print('Writing images to results dir: ' + demo_path_pattern)

    im_names = glob.glob(os.path.join(demo_path_pattern, '*.png')) + \
               glob.glob(os.path.join(demo_path_pattern, '*.jpg'))

    if not (len(im_names) == 0):
        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(('Demo for {:s}'.format(im_name)))
            ctpn(sess, net, im_name)
    else:
        print('No objects detected in any image!')
