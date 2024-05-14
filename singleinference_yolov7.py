
import random
import numpy as np
import os
import sys
import torch
import cv2
import logging

class SingleInference_YOLOV7:
    def __init__(self,
    img_size, path_yolov7_weights, 
    path_img_i='None',
    device_i='cpu',
    conf_thres=0.25,
    iou_thres=0.5):

        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.clicked=False
        self.img_size=img_size

        self.path_yolov7_weights=path_yolov7_weights
        self.path_img_i=path_img_i

        from utils.general import check_img_size, non_max_suppression, scale_coords
        from utils.torch_utils import select_device
        from models.experimental import attempt_load
        self.scale_coords=scale_coords
        self.non_max_suppression=non_max_suppression
        self.select_device=select_device
        self.attempt_load=attempt_load
        self.check_img_size=check_img_size

        self.predicted_bboxes_PascalVOC=[]
        self.im0=None
        self.im=None
        self.device = self.select_device(device_i)
        self.half = self.device.type != 'cpu'
        self.logging=logging
        self.logging.basicConfig(level=self.logging.DEBUG)



    def load_model(self):
        self.model = self.attempt_load(self.path_yolov7_weights, map_location=self.device)
        self.stride = int(self.model.stride.max()) 
        self.img_size = self.check_img_size(self.img_size, s=self.stride)
        if self.half:
            self.model.half() # to FP16

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[139,0,0] for _ in self.names]

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once

    def read_img(self,path_img_i):
        if type(path_img_i)==type('string'):
            if os.path.exists(path_img_i):
                self.path_img_i=path_img_i
                self.im0=cv2.imread(self.path_img_i)
                print('self.im0.shape',self.im0.shape)
            else:
                log_i=f'read_img \t Bad path for path_img_i:\n {path_img_i}'
                self.logging.error(log_i)
        else:
            log_i=f'read_img \t Bad type for path_img_i\n {path_img_i}'
            self.logging.error(log_i)


    def load_cv2mat(self,im0=None):
        if type(im0)!=type(None):
            self.im0=im0
        if type(self.im0)!=type(None):
            self.img=self.im0.copy()    
            self.imn = cv2.cvtColor(self.im0, cv2.COLOR_BGR2RGB)
            self.img=self.imn.copy()
            image = self.img.copy()
            image, self.ratio, self.dwdh = self.letterbox(image,auto=False)
            self.image_letter=image.copy()
            image = image.transpose((2, 0, 1))

            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)
            self.im = image.astype(np.float32)
            self.im = torch.from_numpy(self.im).to(self.device)
            self.im = self.im.half() if self.half else self.im.float()
            self.im /= 255.0
            if self.im.ndimension() == 3:
                self.im = self.im.unsqueeze(0)
        else:
            log_i=f'load_cv2mat \t Bad self.im0\n {self.im0}'
            self.logging.error(log_i)


    def inference(self):
        if type(self.im)!=type(None):
            self.outputs = self.model(self.im, augment=False)[0]
            # Apply NMS
            self.outputs = self.non_max_suppression(self.outputs, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
            img_i=self.im0.copy()
            self.ori_images = [img_i]
            self.predicted_bboxes_PascalVOC=[]
            for i,det in enumerate(self.outputs):
                if len(det):
                    batch_id=i
                    image = self.ori_images[int(batch_id)]

                    for j,(*bboxes,score,cls_id) in enumerate(reversed(det)):
                        x0=float(bboxes[0].cpu().detach().numpy())
                        y0=float(bboxes[1].cpu().detach().numpy())
                        x1=float(bboxes[2].cpu().detach().numpy())
                        y1=float(bboxes[3].cpu().detach().numpy())
                        self.box = np.array([x0,y0,x1,y1])
                        self.box -= np.array(self.dwdh*2)
                        self.box /= self.ratio
                        self.box = self.box.round().astype(np.int32).tolist()
                        cls_id = int(cls_id)
                        score = round(float(score),3)
                        name = self.names[cls_id]
                        self.predicted_bboxes_PascalVOC.append([name,x0,y0,x1,y1,score])
                        color = self.colors[self.names.index(name)]
                        print(20*"*")
                        print(type(score))
                        print(name)
                        name = str(name)+' ' + str(score)
                        cv2.rectangle(image,self.box[:2],self.box[2:],color,2)
                        cv2.putText(image,name,(self.box[0], self.box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[0, 0, 0],thickness=3)
                    self.image=image
                else:
                    self.image=self.im0.copy()
        else:
            log_i=f'Bad type for self.im\n {self.im}'
            self.logging.error(log_i)

    def show(self):
        if len(self.predicted_bboxes_PascalVOC)>0:
            self.TITLE='Press any key or click mouse to quit'
            cv2.namedWindow(self.TITLE)
            cv2.setMouseCallback(self.TITLE,self.onMouse)
            while cv2.waitKey(1) == -1 and not self.clicked:
                cv2.imshow(self.TITLE, self.image)
            cv2.destroyAllWindows()
            self.clicked=False
        else:
            log_i=f'Nothing detected for {self.path_img_i} \n \t w/ conf_thres={self.conf_thres} & iou_thres={self.iou_thres}'
            self.logging.debug(log_i)

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)
    def onMouse(self,event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONUP:
            self.clicked=True

if __name__=='__main__':  
    pass
