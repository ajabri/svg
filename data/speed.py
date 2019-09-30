import os
import io
import numpy as np
from PIL import Image

import cv2
from scipy.ndimage import gaussian_filter
# from scipy.misc import imread 


class NeedForSpeed(object):
    
    """Data Handler that loads need for speed data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64):
        self.root_dir = data_root 
        self.is_train = train
        self.image_size = image_size

        if train:
            self.data_dir = '%s' % self.root_dir
            self.ordered = False
        else:
            self.data_dir = '%s' % self.root_dir
            self.ordered = True 

        self.dirs = open('%s/240fps_dirs.txt' % self.root_dir,'r').readlines()
        self.dirs = [os.path.join(self.root_dir, d.strip()) for d in self.dirs]

        # for d1 in os.listdir(self.data_dir):
        #     for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
        #         self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))

        self.seq_len = seq_len
        self.image_size = image_size 
        self.seed_is_set = False # multi threaded loading
        self.d = 0

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return 10000

    def pick_region(self, x1, x2):
        import time
        t0 = time.time()
        sz = self.image_size // 2

        s1, s2 = (x[sz:-sz, sz:-sz] for x in (x1, x2))
        s1, s2 = (cv2.cvtColor(np.array(x), cv2.COLOR_BGR2GRAY) for x in (s1, s2))
        flow = cv2.calcOpticalFlowFarneback(s1, s2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        t1 = time.time()
        blur_flow = gaussian_filter(np.linalg.norm(flow, axis=-1), sigma=10)
        t2 = time.time()

        
        x, y = np.unravel_index(np.argmax(blur_flow, axis=None), blur_flow.shape)
        t3 = time.time()

        # print(t1 - t0, 'flow took')
        # print(t2 - t1, 'blur took')

        return (x, x + 2*sz, y, y + 2*sz)
        
    def get_seq(self):
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]

        image_seq = []

        start = np.random.randint(0, int(len(os.listdir(d))/2 - self.seq_len))
        # print('start', start, list(range(start, start + self.seq_len)))

        flowers = []
        sz = self.image_size * 5
        for i in range(start, start + 2):
            i = 2*i + 2 if self.is_train else 2*i + 1
            fname = '%s/%05d.jpg' % (d, i)

            im = cv2.resize(cv2.imread(fname), (sz, sz))#.reshape(1, sz, sz, 3)
            flowers.append(im)

        x1, x2, y1, y2 = self.pick_region(*flowers)

        for i in range(start, start + self.seq_len):
            i = 2*i + 2 if self.is_train else 2*i + 1
            fname = '%s/%05d.jpg' % (d, i)
            
            # print(x1, x2, y1, y2)
            im = cv2.resize(cv2.imread(fname), (sz, sz)).reshape(1, sz, sz, 3)
            im = im[:, x1:x2, y1:y2]

            image_seq.append(im/255.)

        image_seq = np.concatenate(image_seq, axis=0) #.transpose(0, 3, 1,2)

        return image_seq


    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()


