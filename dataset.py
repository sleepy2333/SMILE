import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle



class mmDataset(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        assert data in ['mosi', 'mosei'] and split_type in ['train', 'valid', 'test']
        super().__init__()
        #assert dataset_path == '/home/qsw/Self-MM-main/datasets/MOSEI/aligned_50.pkl'
        dataset_path = os.path.join(dataset_path, data.upper(), 'aligned_50.pkl' if if_align else 'unaligned_50.pkl' )

        dataset     = pickle.load(open(dataset_path, 'rb'))
        self.text   = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio  = torch.tensor(dataset[split_type]['audio'].astype(np.float32)).cpu().detach()
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()        
        self.labels = torch.tensor(dataset[split_type]['regression_labels'].astype(np.float32)).cpu().detach()
        self.labels = self.labels#.unsqueeze(-1).unsqueeze(-1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        L = self.text[index]
        A = self.audio[index]
        V = self.vision[index]
        Y = self.labels[index]
        return L, A, V, Y

class MMDataset(Dataset):
    def __init__(self, dataset_path, data='mosei', split_type='train', if_align=False):
        assert data in ['mosi', 'mosei', 'iemocap'] and split_type in ['train', 'test', 'valid']
        super().__init__()
        data = 'mosei_senti' if data == 'mosei' else data
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))
        print('load success!')

        self.data = data
        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text   = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio  = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio  = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach().squeeze(1).squeeze(1)
        #self.meta   = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
        

    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, index):
        L = self.text[index]
        A = self.audio[index]
        V = self.vision[index]
        Y = self.labels[index]
        '''
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META
        '''
        return L, A, V, Y


"""
CMU-MOSEI info
Train 16326 samples
Val 1871 samples
Test 4659 samples
CMU-MOSEI feature shapes
visual: (60, 35)
audio: (60, 74)
text: GLOVE->(60, 300)
label: (6) -> [happy, sad, anger, surprise, disgust, fear] 
    averaged from 3 annotators
unaligned:
text: (50, 300)
visual: (500, 35)
audio: (500, 74)    
"""

emotion_dict = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5}


class AlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type
        self.visual, self.audio, \
            self.text, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type):
        try:
            data = torch.load(self.data_path)
        except:
            data = pickle.load(open(self.data_path, 'rb'))

        data = data[data_type]
        visual = data['src-visual']
        audio = data['src-audio']
        text = data['src-text']
        labels = data['tgt']
        return visual, audio, text, labels

    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]

        text_mask = np.array(text_mask)

        return text, text_mask

    def _get_visual(self, index):
        visual = self.visual[index]
        visual_mask = [1] * visual.shape[0]
        visual_mask = np.array(visual_mask)

        return visual, visual_mask

    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = [1] * audio.shape[0]

        audio_mask = np.array(audio_mask)

        return audio, audio_mask

    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emotion_dict[emo]] = 1

        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)

        return labels_embedding, labels_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, audio, audio_mask, \
            visual, visual_mask, label


class UnAlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type
        self.visual, self.audio, \
            self.text, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type):
        label_data = torch.load(self.data_path)
        label_data = label_data[data_type]
        with open('data/mosei_senti_data_noalign.pkl', 'rb') as f:
            data = pickle.load(f)
        data = data[data_type]
        visual = data['vision']
        audio = data['audio']
        text = data['text']
        audio = np.array(audio)
        labels = label_data['tgt']
        return visual, audio, text, labels

    def _get_text(self, index):
        text = self.text[index].astype(np.float)
        text_mask = [1] * text.shape[0]

        text_mask = np.array(text_mask)

        return text, text_mask

    def _get_visual(self, index):
        visual = self.visual[index].astype(np.float)
        visual_mask = [1] * 50
        visual_mask = np.array(visual_mask)

        return visual, visual_mask

    def _get_audio(self, index):
        audio = self.audio[index].astype(np.float)
        audio[audio == -np.inf] = 0
        audio_mask = [1] * 50

        audio_mask = np.array(audio_mask)

        return audio, audio_mask

    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emotion_dict[emo]] = 1

        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)

        return labels_embedding, labels_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, audio, audio_mask, \
            visual, visual_mask, label

