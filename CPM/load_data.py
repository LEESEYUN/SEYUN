from pycocotools.coco import COCO
import tensorflow as tf
import os, sys
from utils import dataset_util

img_dir='/data/COCO/2014_2015/'
ann_dir='/data/COCO/2014_2015/annotations'
#annTypes={'instances', 'captions', 'person_keypoints'}
tfr_dir='/data/COCO/2014_2015/tfrecord/'

annTypes='person_keypoints'
annType=annTypes
dataType = 'train2014'
annFile=ann_dir+'/%s_%s.json'%(annType,dataType)




flags = tf.app.flags
flags.DEFINE_string('data_dir', img_dir, 'Root directory to raw Microsoft COCO dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set or validation set')
flags.DEFINE_string('output_filepath',tfr_dir+'train2014/test.tfrecord', 'Path to output TFRecord')
flags.DEFINE_bool('shuffle_imgs',True,'whether to shuffle images of coco')
FLAGS = flags.FLAGS


def load_coco_dection_dataset(imgs_dir, annotations_filepath, shuffle_img = True ):
    coco = COCO(annFile)

    ann_ids=coco.getAnnIds()
    cat_ids=coco.getCatIds()
    img_ids=coco.getImgIds()


    #print(len(ann_ids))
    print(cat_ids)
    fns= len(img_ids)

    nb_imgs = len(img_ids)
    coco_data = []

    for index, img_id in enumerate(img_ids):
        if index >500: break;
        if index % 100 == 0:
            print("Readling images: %d / %d " % (index, nb_imgs))
        img_info = {}
        bboxes = []
        labels = []

        img_detail = coco.loadImgs(img_id)[0]
        pic_height = img_detail['height']
        pic_width = img_detail['width']
        image_name= [img_id]
        keypoints =[]
        num_keypoints=[]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        print(anns[0]['image_id'])
        print(anns[1]['image_id'])
        for ann in anns:

            #   print("\n")
            keypoints_data=ann['keypoints']


            keypoints.append(keypoints_data)
            num_keypoints_data=ann['num_keypoints']
            num_keypoints.append(num_keypoints_data)
            bboxes_data = ann['bbox']
            bboxes_data = [bboxes_data[0] / float(pic_width), bboxes_data[1] / float(pic_height), \
                            bboxes_data[2] / float(pic_width), bboxes_data[3] / float(pic_height)]
        # the format of coco bounding boxs is [Xmin, Ymin, width, height]
            bboxes.append(bboxes_data)
            labels.append(ann['category_id'])

        img_path = os.path.join(imgs_dir, img_detail['file_name'])
        img_bytes = tf.gfile.FastGFile(img_path, 'rb').read()

        img_info['pixel_data'] = img_bytes
        img_info['height'] = pic_height
        img_info['width'] = pic_width
        img_info['bboxes'] = bboxes
        img_info['labels'] = labels
        img_info['keypoints'] = keypoints
        img_info['num_keypoints'] = num_keypoints
        img_info['img_id']=image_name
        print(img_info.get('img_id'))
        coco_data.append(img_info)
    return coco_data
def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])
    #'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder'\
    #'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    key_p=[]
    keypoints=img_data['keypoints']
    for keypoint in keypoints:
      #  for temp in range(51):
            key_p.append(keypoint)

    #num_key_p=[]
    num_keypoints=img_data['num_keypoints']
    len_numkey=len(num_keypoints)
    num_key_p=[len_numkey]
    #print(img_data['img_id'])





    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_data['height']])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_data['width']])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=img_data['labels'])),
        'image/keypoints' : tf.train.Feature(int64_list=tf.train.Int64List(value=key_p)),
        'image/num_keypoints' : tf.train.Feature(int64_list=tf.train.Int64List(value=num_key_p)),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data['pixel_data']])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf-8')]))
    }))
    return example

def main(_):
    if FLAGS.set == "train":
        imgs_dir = os.path.join(FLAGS.data_dir, 'train2014')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','person_keypoints_train2014.json')
        print("Convert coco train file to tf record")
    elif FLAGS.set == "val":
        imgs_dir = os.path.join(FLAGS.data_dir, 'val2014')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','person_keypoints_val2014.json')
        print("Convert coco val file to tf record")
    else:
        raise ValueError("you must either convert train data or val data")
    # load total coco data
    coco_data = load_coco_dection_dataset(imgs_dir,annotations_filepath,shuffle_img=FLAGS.shuffle_imgs)
    total_imgs = len(coco_data)
    # write coco data to tf record
    with tf.python_io.TFRecordWriter(FLAGS.output_filepath) as tfrecord_writer:
        for index, img_data in enumerate(coco_data):
            if index % 100 == 0:
                print("Converting images: %d / %d" % (index, total_imgs))
            example = dict_to_coco_example(img_data)
            tfrecord_writer.write(example.SerializeToString())


if __name__ == "__main__":
    tf.app.run()


#print(coco)
#for fn in range(1,my_anno)