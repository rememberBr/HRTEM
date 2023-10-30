import numpy as np
import cv2
import pdb
import os

from unet3080 import *
from data import *

from skimage import measure
import math
import copy

import itertools
import json
import re

class PDF():
    def __init__(self):
        self.colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (0,0,127), (0,127,0), (127,0,0), (0,127,127), (127,127,0), (127,0,127)]
        self.PDFdata = {}
        self.PDFlist = []
        self.matchlist = []
        self.color = {}
        self.colorid = 0
        self.res = {}

    def readPDF(self, path):
        tmp = []
        with open(path, "r") as f:
            need = False
            for line in f.readlines():
                if line[:4] == 'd(A)':
                    need = True
                if need:
                    line = line.strip('\n')
                    tmp.append([i for i in line.split(' ') if i != ''])
        new = [[i[0], i[3],i[4],i[5]] for i in tmp[1:]] 
        return new


    def angle(self, x, y):
        x = np.array(x)
        y = np.array(y)

        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))

        cos = x.dot(y)/(Lx*Ly)
        ang = round(np.arccos(cos)*360/2/np.pi, 2)
        return ang

    def loadPDF(self):

        for root, non, files in os.walk('PDF/'):
            pass
        for obj in files:
            path = 'PDF/'+obj
            reader = self.readPDF(path)
            dic = {i[0]:i[1:] for i in reader}
            id = [i[0] for i in reader]
            combination = []
            for i in itertools.permutations(id, 2):
                combination.append(i)
            newData = []
            for i in range(len(combination)):
                x = [int(j) for j in dic[combination[i][0]]]
                y = [int(j) for j in dic[combination[i][1]]]
                ang = self.angle(x, y)
                if i < len(id):
                    newData.append([combination[i], ang, id[i], np.cross(x,y)])
                else:
                    newData.append([combination[i], ang, '', np.cross(x,y)])
            name = re.split(r'[\\/]', path.split('.')[0])[-1]
            self.PDFdata[name] = newData  
            self.PDFlist.append(name)
   
    def chosePDF(self, Crystals, name):
        colorid = 0
        paths = name.split(' ')
        for i in paths:
            Crystals.args.PDFs[i] = self.PDFdata[i]
            Crystals.args.matchlist.append(i)
            Crystals.args.color[i] = self.colors[colorid]
            colorid += 1
        return 1



def load_unet():
    myunet = myUnet()
    model = myunet.get_unet_lca()
    model.load_weights('lcaunet.h5')
    model.predict(np.ones((1,64,64,1))/2, verbose=1)
    return model

def imgSplit(args, img, grid_h, step):
    # pdb.set_trace()
    args.x1 = 0
    args.y1 = 0
    args.x2 = img.shape[1]
    args.y2 = img.shape[0]
    w = args.x2-args.x1
    h = args.y2-args.y1
    if h % grid_h != 0:
        new_y = args.y2 + (grid_h - h % grid_h)
        
        if new_y > img.shape[0]:
            new_y = args.y1 - (grid_h - h % grid_h)
            if new_y<0:
                args.y1 += (h % grid_h)
            else:
                args.y1 -= (grid_h - h % grid_h)
        else:
            args.y2 += (grid_h - h % grid_h)
    if w % grid_h != 0:
        new_x = args.x2 + (grid_h - w % grid_h)
        if new_x > img.shape[0]:
            new_x = args.x1 - (grid_h - w % grid_h)
            if new_x<0:
                args.x1 += (w % grid_h)
            else:
                args.x1 -= (grid_h - w % grid_h)
        else:
            args.x2 += (grid_h - w % grid_h)
    num_w = (((args.x2-args.x1-grid_h)//step)+1)
    num_h = (((args.y2-args.y1-grid_h)//step)+1)
    return num_w, num_h, num_w*num_h 




def divide_method(img, grid_h, d):#分割成m行n列
    h, w = img.shape[0], img.shape[1]
    m, n, step = h/d, w/d, grid_h/d
    if m%1 !=0 or n%1 !=0 or step%1 != 0:
        print("不支持的尺寸，可能需要修改或者将原图resize")
        return -1
    else:
        m, n, step = int(h/d), int(w/d), int(grid_h/d)
    gx, gy = np.meshgrid(np.linspace(0, w, n+1),np.linspace(0, h, m+1))
    gx=gx.astype(np.int)
    gy=gy.astype(np.int)
    divide_image = np.zeros([m-step+1, n-step+1, grid_h, grid_h], np.uint8)#这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息 
    for i in range(m-step+1):
        for j in range(n-step+1):   
            divide_image[i,j]=img[gy[i][j]:gy[i+step][j+step], gx[i][j]:gx[i+step][j+step]]
    FFTs = FFT(divide_image,grid_h)
    return FFTs


def FFT(divide_image, grid_h):
    m,n=divide_image.shape[0], divide_image.shape[1]
    FFTs = []
    step = (grid_h-64)//2
    for i in range(m):
        for j in range(n):
            tt = divide_image[i,j,:]
            tt = tt.astype('float64')/255.0
            dft = cv2.dft(tt, flags = cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            f1 = np.log(cv2.magnitude(dft_shift[:, :, 0],dft_shift[:, :, 1])+1)
            out = np.zeros(f1.shape, np.double)
            magnitude_spectrum = cv2.normalize(f1, out, 1.0, 0.0, cv2.NORM_MINMAX)
            FFTs.append(magnitude_spectrum[step:step+64, step:step+64]*255)
    return FFTs

        
def load_data(FFTs):
    imgs_pre = np.ndarray((len(FFTs), 64, 64, 1), dtype=np.float32)
    for i in range(len(FFTs)):
        im = FFTs[i].reshape((64, 64, 1)).astype('float32')
        imgs_pre[i] = (im/255)**2.5*255
    return imgs_pre.astype('uint8')


#计算夹角
def angle(v1, v2):
    dx1 = v1[0]
    dy1 = v1[1]
    dx2 = v2[0]
    dy2 = v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = angle1 * 180/math.pi
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = angle2 * 180/math.pi
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

#约等于
def Aequal(num1, num2, dist_error, angle_error):
    if type(num1) == np.ndarray or type(num1) == list:
        if abs(num1[0]-num2[0])<dist_error and abs(num1[1]-num2[1])<dist_error:
            return True
        else:
            return False
    else:
        if abs(num1-num2)<angle_error:
            return True
        else:
            return False

#计算点距离 
def Distance(point1,point2):
    vec1 = np.array(point1)
    vec2 = np.array(point2)
    distance = np.linalg.norm(vec1 - vec2)
    return distance

#判断是否属于同一条直线
def isBoomerang(points, dist_error, angle_error):
    x1, y1 = points[0][0], points[0][1]
    x2, y2 = points[1][0], points[1][1]
    x3, y3 = points[2][0], points[2][1]
    return Aequal((x3 - x1) * (y2 - y1), (x2 - x1) * (y3 - y1), dist_error, angle_error)

#重心法求斑点中心
def point_center(img, image_np, name):
    img = (255-img*255)/255
    mask = img>0.5
    label_image, num = measure.label(mask, connectivity = 1, return_num = True)
    label = {i:[] for i in range(1, num+1)}
    for i in range(1, num+1):
        x_y = np.where(label_image == i)
        label[i].append(list(zip(x_y[0], x_y[1])))                
    vecter = []

    for i in range(1,num+1):
        tmp = np.array(label[i][0])
        x = tmp[:,0]
        y = tmp[:,1]
        imtmp = []
        for j in tmp:
            imtmp.append(img[j[0],j[1]])
        w1 = imtmp/sum(imtmp)
        aa = np.matrix(w1)
        w2 = aa*tmp

        if abs(w2[0, 1] - 32) < 1 and abs(w2[0, 0] - 32) < 1: #去除中心点
            continue
        else:
            vecter.append([w2[0, 1] - 32, w2[0, 0] - 32])
    vecter = np.array(vecter)
    return vecter

#去除关于中心对称的点
def clear_symmetry(vecter, image_np, name, dist_error, angle_error):
    del_idx = []
    
    for num in range(len(vecter)):
        sta = False #是否存在对称点
        for num1 in range(num+1,len(vecter)):
            if Aequal(vecter[num], -vecter[num1], dist_error, angle_error):
                sta = True
                if vecter[num][1]<=vecter[num1][1]:
                    del_idx.append(num1)
                else:
                    del_idx.append(num)
        #无对称也删除  模型质量非常好不一定需要这个！！！，看情况是否添加
        if not sta:
            del_idx.append(num)
    del_idx = list(set(del_idx))
    del_idx.sort(reverse=True)
    for d in del_idx:
        vecter = np.delete(vecter, d, axis=0)  
    return vecter

#清除延长线上距离中心最远的点
def clear_extension_cord(vecter, image_np, name, dist_error, angle_error):
    del_idx = []
    for num in range(len(vecter)):
        for num1 in range(num+1,len(vecter)):
            if isBoomerang([[32, 32], [vecter[num, 0]+32, vecter[num, 1]+32], [vecter[num1][0]+32, vecter[num1][1]+32]], dist_error, angle_error):
                if np.linalg.norm(vecter[num]) <= np.linalg.norm(vecter[num1]):
                    del_idx.append(num1)
                else:
                    del_idx.append(num)
    del_idx = list(set(del_idx))
    del_idx.sort(reverse=True)

    for d in del_idx:
        vecter = np.delete(vecter, d, axis=0)
    return vecter





#寻找最终存在的平行四边形的数据
def parallelogram_data(magnification, pre_res, PDF_data, dist_error, angle_error, PDF_dist_error, PDF_angle_error):
    all_data = {}
    name = 0
    dists = {}
    angs = {}
    orientations = {}
    # pdb.set_trace()
    # 下面这点东西是通用的，也许可以抽出来，不需要每次都处理pdf卡片
    for k, v in PDF_data.items():
        dists[k] = [i[2] for i in v if i[2] != '']
        angs[k] = {i[0]:i[1] for i in v}
        orientations[k] = {i[0]:i[3] for i in v}
    # pdb.set_trace()
    for im in pre_res:
        # pdb.set_trace()
        im = im.reshape(64,64)
        image_np = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        vecter = point_center(im, image_np, name)  
        #寻找关于中心的对称的斑点，若存在，删除靠下面的一个点
        # pdb.set_trace()
        vecter = clear_symmetry(vecter, image_np, name, dist_error, angle_error)
        #清除延长线上距离中心最远的点
        vecter = clear_extension_cord(vecter, image_np, name, dist_error, angle_error)
        minDists = {k:[] for k, v in dists.items()}
        if len(vecter) <2:
            pass
        else:
            dist = [[10/(np.linalg.norm(i)*magnification),i] for i in vecter] #计算每个点到中心的距离 
            for i in dist:
                for ke, va in dists.items():
                    dis = [[j,i[1]] for j in va if (1+PDF_dist_error)*float(j)>i[0] and (1-PDF_dist_error)*float(j)<i[0]]
                    if dis:
                        minDists[ke].extend(dis)
        all_data[name] = copy.deepcopy(minDists)
        name += 1
    tmp = []
    # pdb.set_trace()
    for k, v in all_data.items():
        for ke, va in v.items(): 
            if len(va)>=2:
                tmp.append([ke, va, k])
    # pdb.set_trace()
    res = {}
    for i in tmp:
        key = i[2]
        comb = [j for j in itertools.combinations(i[1], 2)]
        comb1 = [(j[0][0],j[1][0],str(orientations[i[0]][(j[0][0],j[1][0])])) for j in comb if j[0][0]!=j[1][0]\
                and angs[i[0]][(j[0][0],j[1][0])]*(1 + PDF_angle_error)>angle(j[0][1],j[1][1]) \
                and angs[i[0]][(j[0][0],j[1][0])]*(1 - PDF_angle_error)<angle(j[0][1],j[1][1])]    
        if comb1 !=[]:
            try:
                res[key][i[0]] = comb1
            except KeyError:
                res[key] = {i[0]:comb1}
    # pdb.set_trace()
    return res



def allbound(Crystals, PDFnames): 

    #标记参数
    for path, res in Crystals.res.items():
        image_np = cv2.imread(path)
        image_nps = {k : np.ones(image_np.shape, dtype = 'uint8')*255 for k in PDFnames}
        # image_nps = {k.split('/')[-1]:np.ones(image_np.shape,dtype = 'uint8')*255 for k in PDFnames}
        for k,v in res.items():
            # pdb.set_trace()
            #标记参数
            y = int(k) % Crystals.args.img_num_h
            x = int(k) // Crystals.args.img_num_w
            for ke, va in v.items():
                if ke in PDFnames and va !=[]:
                    #标记参数
                    # pdb.set_trace()
                    image_nps[ke][Crystals.args.y1 + x*Crystals.args.step:Crystals.args.y1 + x*Crystals.args.step+256,
                                    Crystals.args.x1 + y*Crystals.args.step:Crystals.args.x1 + y*Crystals.args.step+256,:]=20

        for k,v in image_nps.items():
            thresh = cv2.Canny(v, 128, 256)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.cv2.CHAIN_APPROX_NONE)
            #标记参数
            # pdb.set_trace()
            cv2.drawContours(image_np,contours, -1, Crystals.args.color[k],10)
        # pdb.set_trace()
        cv2.imwrite("res/"+Crystals.args.experiment_id+ "/"+ Crystals.args.args_id+"/"+path.split('/')[-1].split('.')[0]+'.jpg', 
                                                        cv2.resize(image_np, (1024,1024), interpolation=cv2.INTER_AREA))
    return 1

def allarea(Crystals): 
    try:
        areas = {}
        nums = {}
        initalize = False
        Crystals.args.stop = True
        #标记参数
        for path, res in Crystals.res.items():
            if Crystals.args.stop:
                if not initalize:
                    image_np = cv2.imread(path,0)
                    # image_np = dm4.dm4reader(path)
                    image_info = image_np.shape
                    initalize = True
                area = {}
                num = {}
                image_nps = {k : np.zeros(image_info,dtype = 'uint8') for k in Crystals.args.matchlist}
                # image_nps = {k.split('/')[-1]:np.ones(image_info,dtype = 'uint8')*255 for k in PDFnames}
                for k,v in res.items():
                    #标记参数
                    y = int(k)%Crystals.args.img_num_h
                    x = int(k)//Crystals.args.img_num_w
                    for ke, va in v.items():
                        if ke in Crystals.args.matchlist and va !=[]:
                            #标记参数
                            image_nps[ke][Crystals.args.y1 + x*Crystals.args.step:Crystals.args.y1 + x*Crystals.args.step+256,
                                          Crystals.args.x1 +y*Crystals.args.step:Crystals.args.x1 +y*Crystals.args.step+256]=1
                #算面积 有参数
                for k,v in  image_nps.items(): 
                    area[k] = 51.359*51.359*v.sum()/(image_info[0]*image_info[1]) 
                    mask = v>0.5
                    label_image, label_num = measure.label(mask, connectivity = 1, return_num = True)
                    num[k] = label_num
                areas[re.split(r'[\\/]', path.split('.')[0])[-1]] = area
                nums[re.split(r'[\\/]', path.split('.')[0])[-1]] = num
                # areas[path.split('/')[-1].split('.')[0]] = area
            else:
                return -1 
        json_str = json.dumps(areas, indent=4)
        #标记参数
        jsonpath = os.path.join(Crystals.args.respath, 'areas'+'.json')
        with open(jsonpath, 'w') as json_file:
            json_file.write(json_str)

        json_str = json.dumps(nums, indent=4)
        #标记参数
        numpath = os.path.join(Crystals.args.respath, 'nums'+'.json')
        with open(numpath, 'w') as json_file:
            json_file.write(json_str)

        # Crystals.args.res.append(jsonpath)
        # Crystals.logger_model.info('area : {}'.format(jsonpath))
        # Crystals.logger_model.info('num : {}'.format(numpath))

        return os.path.join(os.getcwd(),jsonpath), os.path.join(os.getcwd(),numpath)
    except Exception as e:
        # Crystals.logger_model.error(str(e.__traceback__.tb_lineno)+' '+repr(e))
        return -1