
from func import *

class interfence_args():
    def __init__(self, experiment_id = '',args_id = ''):
        #实验与参数id
        self.experiment_id = experiment_id
        self.args_id = args_id

        #放大率 来自.dm4文件
        self.magnification = 0.311523
        #步长
        self.step = 128
        #误差参数
        #接口 02
        self.dist_error = 2
        #接口 03
        self.angle_error = 20
        #接口 04
        self.PDF_dist_error = 0.05 #0.03
        #接口 20
        self.PDF_angle_error = 0.03
        #patch size
        self.img_num_w = 0 
        self.img_num_h = 0 
        #用于对比的PDF卡片数据
        self.PDFs = {} 
        #每种物质对应的颜色
        self.color = {}
        #PDF的名称
        self.matchlist = []
        #图像路径以及保存路径
        self.imgpath = "aim/"
        self.respath = ""

        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0

class start():
    def __init__(self):
        self.model = load_unet()
        self.PDF = PDF()
        self.args = interfence_args("experiment_id", "args_id")
        self.PDF.loadPDF()
        self.res = {}



Crystals = start()
### args是实验参数，需要设置实验id和参数id
# args = interfence_args("experiment_id", "args_id") #有参数
#加载PDF卡片
Crystals.PDF.chosePDF(Crystals, "PDF#37-1484 PDF#05-0665") #多个卡片中间空格隔开
# pdb.set_trace()
#滑动步长
step = Crystals.args.step        
#本次推理结果
inference_res = {}

#匹配使用的PDF卡片信息
PDF_data = Crystals.args.PDFs

#匹配误差参数
dist_error = Crystals.args.dist_error
angle_error = Crystals.args.angle_error
PDF_dist_error = Crystals.args.PDF_dist_error
PDF_angle_error = Crystals.args.PDF_angle_error

#保存实验结果的路径
Crystals.args.respath = os.path.join('res', Crystals.args.experiment_id, Crystals.args.args_id)

#滑动窗口尺寸
grid_h = 256  #有参数

correction = False
img_num_all = 0

#保存结果
data = {}

history = { "step":step,
            "imgpath" :Crystals.args.imgpath,
            "matchlist":Crystals.args.matchlist,
            "color":Crystals.args.color,
            "experiment_id":Crystals.args.experiment_id,
            "args_id":Crystals.args.args_id,
            "respath":Crystals.args.respath,
            "dist_error":dist_error,
            "angle_error":angle_error,
            "PDF_dist_error":PDF_dist_error,
            "PDF_angle_error":PDF_angle_error
            }


imgpath = []
for root, dirs, files in os.walk(Crystals.args.imgpath):  # 此处有bug  如果调试的数据还放在这里，将会递归的遍历所有文件
    for file in files:
        if os.path.splitext(file)[1] == '.bmp':
            imgpath.append(Crystals.args.imgpath+str(file))   

for i in range(len(imgpath)):
    #可能需要根据自己图片命名的方式进行修改
    name = imgpath[i].split('\\')[-1].split('.')[0]
    path = imgpath[i]
    image = cv2.imread(path,0)
    if image is None:
        print(path+' can not read')
        continue
    #计算滑动窗口的数量
    if  not correction:
        Crystals.args.img_num_w, Crystals.args.img_num_h, img_num_all = imgSplit(Crystals.args, image, grid_h, step)
        history["img_num_w"] = Crystals.args.img_num_w
        history["img_num_h"] = Crystals.args.img_num_h
        correction = True

    FFTs =  divide_method(image, grid_h, step)
    imgs_pre = load_data(FFTs)
    imgs_pre = imgs_pre/255
    pre_res = Crystals.model.predict(imgs_pre, verbose=1)
    inference_res[path] = parallelogram_data(Crystals.args.magnification, pre_res, Crystals.args.PDFs, dist_error, angle_error, PDF_dist_error, PDF_angle_error)

# pdb.set_trace()
#保存结果
data["data"] = inference_res
Crystals.res = inference_res
#创建存放结果的文件夹，实验id/参数id，唯一
try:
    os.makedirs(Crystals.args.respath)
except FileExistsError:
    pass

datapath = os.path.join(Crystals.args.respath, 'data'+'.json')
argspath = os.path.join(Crystals.args.respath, 'history'+'.json')

history['data'] = os.path.join(os.getcwd(),datapath)
json_args = json.dumps(history, indent=4)
json_data = json.dumps(data, indent=4)

with open(datapath, 'w') as json_file:
    if json_file.write(json_data):
        print('save data file success: {}'.format('data'+'.json'))
    
with open(argspath, 'w') as json_file:
    if json_file.write(json_args):
        print('save history file success: {}'.format('history'+'.json'))
    

# Crystals.args.historys[args.experiment_id] = argspath
print(' finished '+ os.path.join(os.getcwd(), argspath))


# print(PDFcomp.loadhistory("res/experiment_id/args_id/history.json", Crystals))
print(allbound(Crystals, ["PDF#37-1484"])) #有参数
print(allarea(Crystals))