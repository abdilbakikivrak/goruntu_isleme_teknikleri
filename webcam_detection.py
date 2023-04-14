import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf

import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from utils import label_map_util

from utils import visualization_utils as vis_util

import cv2

cap = cv2.VideoCapture(0)   

# indirilen model 
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# algılama grafiği için oluşturulan gerçek model
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

#  her tespit kutusuna doğru eşleştirmeyi yapmak için ve nesne belirtmeleri kullanılan dizi listesi 
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ##  gerekli görülürse veri seti çekilebilir *başlangıçta

# In[5]:

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())


# ## tf modelini hafızaya yükleme 

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## etiket haritası ve
#Etiket haritaları dizinleri kategori adlarına eşler,
# böylece model seti ağımız "5"i tahmin ettiğinde bunun "uçak"a karşılık geldiğini biliriz.
# Burada dahili yardımcı işlevler kullanıyoruz,
# ancak bir sözlük eşleme tamsayılarını uygun dize etiketlerine döndüren veri seti oluşturmak iyi olur.

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ##  veri seti eşleştirmede yardımcı kod

# In[8]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # yakalama ve tespit

# In[9]:

# basitlik için iki resim kullanılacak olursa
# image1.jpg
# image2.jpg
# Kodu resimlerle test etmek istersek, resimlerin yolunu TEST_IMAGE_PATHS'e eklememiz yeterlidir..
#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]  # change this value if you want to add more pictures to test

# çıktı görüntülerinin inc cinsinden boyutu
#IMAGE_SIZE = (12, 8)


# In[10]:

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            # modelin verdiği çıktı için boyutları beklenen gibi genişletmek daha uyumlu olur: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            #ekran çıktısında her yakalama için bir kutu içerisinde nesnenin algılandığı bölümünü işaret edeer
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # yüzdelik değer yakalanan nesnenin eşleştirilme oranınnı ifade eder
            # eşleştirilme oranı cinsi ile çıktı ekranında etiketlenir
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # algılama 
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # algılamanın sonuçlarının görselleştirilmesi
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
