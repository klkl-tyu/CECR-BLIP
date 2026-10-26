import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
import re
from dataset.video_transform import *
import numpy as np
import cv2
import json
import random
def safe_read_frame(cap, frame_num, total_frames):
    for offset in range(1, 1000):  # 尝试 ±1 ~ ±4 帧
        for new_frame in [frame_num - offset, frame_num + offset]:
            if 0 <= new_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                ret, frame = cap.read()
                if ret:
                    return frame
    return None
def get_clip_video_frames(video_path, processor, num, is_counterfactual=False, box = None): #
    cap = cv2.VideoCapture(video_path)
    #print(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = 4
    sample_interval = total_frames // sample_frames
    imgs = []
    j = int(num/25)
    for i in range(sample_frames):
        frame_num = i * sample_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, cv2_im = cap.read()
        ### 在这里加mask吧
        if is_counterfactual:
            cv2.rectangle(cv2_im, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 0), -1)
            # cv2.imwrite('masked_image.jpg', cv2_im)
        
        if ret:
            converted = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(converted)
            if i == j:
                imgs.append(pil_im)
        else:
            if i == j:
                cv2_im = safe_read_frame(cap, frame_num, total_frames)
                if cv2_im is not None:
                    if is_counterfactual:
                        cv2.rectangle(cv2_im, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 0), -1)
                    converted = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                    pil_im = Image.fromarray(converted)
                    imgs.append(pil_im)
    cap.release()
    
    # # prompt = "The video's emotion is "
    ### 或者在这个前面加？这里是PIL格式的了
    images = processor(images=imgs, return_tensors="pt")
    # # images = torch.cat([clip_preprocess(x) for x in imgs])
    #images["pixel_values"] = images["pixel_values"].mean(dim=0)
    # images['input_ids'] = images['input_ids'].squeeze(0)
    # images['attention_mask'] = images['attention_mask'].squeeze(0)
        
    return images

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data.split(' ')[0]

    @property
    def label_text(self):
        return self._data.split(' ')[1]

    @property
    def label(self):
        return int(self._data.split(' ')[2])
    
class VideoDataset(data.Dataset):
    def __init__(self, list_file, mode, transform, facebox_data = None, gt_caption = None):
        self.list_file = list_file
        self.mode = mode
        self.transform = transform
        self.caption = gt_caption
        self.facebox_data = facebox_data
        with open('mafw/mafw_train_facebox_newnew14.json', 'r', encoding='utf-8') as file:
            facebox_data14 = json.load(file)    
        pass
        self.facebox_data14 = facebox_data14
    def _parse_list(self):
        self.video_list = [VideoRecord(item) for item in self.list_file]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # split all frames into seg parts, then select frame in each part randomly
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        # split all frames into seg parts, then select frame in the mid of each part
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        record = self.list_file[index]
        return self.get(record)

    def get(self, record):
        path = record
        if self.mode == 'train':
            vpath = path.split(' ')[0].split('.')[0].split('/')[-1]
            num = int(path.split(' ')[1])
            if num == 25:
                face_box = self.facebox_data14[vpath]["phrases"]
                caption = self.facebox_data14[vpath]["caption"]
            else: 
                face_box = self.facebox_data[vpath]["phrases"]
                caption = self.facebox_data[vpath]["caption"]
            if len(face_box) == 1:
                facedata = face_box[0]
            else:
                rand_int = random.randint(0, len(face_box)-1)
                facedata = face_box[rand_int]
            boxs = facedata["boxes"][0]
            video_frames_counterfactual = get_clip_video_frames(path.split(' ')[0], self.transform, num, True, boxs) #self.transform, oricaption
            emotion_captions_counterfactual = facedata["caption"]             

            video_frames = get_clip_video_frames(path.split(' ')[0], self.transform, num) #self.transform, oricaption
            emotion_captions = caption
            
            return (video_frames,video_frames_counterfactual), [emotion_captions, emotion_captions_counterfactual], facedata # record.label_text, oricaption
        else:
            video_frames = get_clip_video_frames(path, self.transform, 50)
            return video_frames, path.split('.')[0].split('/')[-1]
        

    def __len__(self):
        return len(self.list_file)
    
    
def train_data_loader(train_data_list, train_transforms, facebox_data, gt_caption):
    train_data = VideoDataset(list_file=train_data_list,
                              mode='train',
                              transform=train_transforms,
                              facebox_data = facebox_data,
                              gt_caption = gt_caption)
    return train_data


def test_data_loader(test_data_list, test_transforms, gt_caption):
    test_data = VideoDataset(list_file=test_data_list,
                              mode='test',
                              transform=test_transforms,
                              gt_caption = gt_caption)
    return test_data
