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

def get_clip_video_frames(video_path, processor, is_counterfactual=False, box = None): #
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = 4
    sample_interval = total_frames // sample_frames
    imgs = []

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
            imgs.append(pil_im)

    cap.release()
    
    # # prompt = "The video's emotion is "
    ### 或者在这个前面加？这里是PIL格式的了
    images = processor(images=imgs, return_tensors="pt")
    # # images = torch.cat([clip_preprocess(x) for x in imgs])
    images["pixel_values"] = images["pixel_values"].mean(dim=0)
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
            
        pass

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
        caption = self.caption[record.split('.')[0].split('/')[-1]]
        return self.get(record, caption)

    def get(self, record,caption):
        path = record
        sentences = re.split(r'[.。]', caption)
        if self.mode == 'train':
            face_box = self.facebox_data[path.split('.')[0].split('/')[-1]]["phrases"][0]["boxes"][0]
            face_data = self.facebox_data[path.split('.')[0].split('/')[-1]]
            video_frames_counterfactual = get_clip_video_frames(path, self.transform, True, face_box) #self.transform, oricaption
            emotion_captions_counterfactual = sentences[0]+'.'             

            video_frames = get_clip_video_frames(path, self.transform) #self.transform, oricaption
            emotion_captions = caption
            
            return (video_frames,video_frames_counterfactual), [emotion_captions, emotion_captions_counterfactual], face_data # record.label_text, oricaption
        else:
            video_frames = get_clip_video_frames(path, self.transform)
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
