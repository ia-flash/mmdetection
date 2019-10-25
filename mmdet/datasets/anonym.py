import mmcv
import json
from .custom import CustomDataset
import  numpy as np
class AnonymDataset(CustomDataset):

    CLASSES = ('person','plate')

    def __init__(self, **kwargs):
        super(AnonymDataset, self).__init__(**kwargs)

    def get_ann_info(self, idx):
        img = self.img_infos[idx]
        #print(img)
        ann = self.img_infos[idx]['ann']
        bboxes = ann.get('bboxes',None)
        bboxes_ignore = ann.get('bboxes_ignore',None)
        labels = ann['labels']
        labels_ignore =  ann.get('labels_ignore',None)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        """
        print("max label : %s"%max(labels.astype(np.int64)))
        print(bboxes.astype(np.float32).shape)
        print("max bboxes X : %s"%bboxes.astype(np.float32)[:,2].max())
        print("max bboxes Y : %s"%bboxes.astype(np.float32)[:,3].max())
        print("H : %s"%self.img_infos[idx]['width'] )
        print("W : %s"%self.img_infos[idx]['height'])
        """
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
