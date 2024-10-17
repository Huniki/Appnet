import math
import torch
import torch.nn as nn
 
from lib.models.common import DyHeadBlock,Conv, DFL,dist2bbox, make_anchors


 

class Detect_DyHead(nn.Module):
    """YOLOv8 Detect head with DyHead for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    #  # init
    # anchors = [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]]  # init
    anchors = torch.empty(0) 
    strides = torch.empty(0)  # init
 
    def __init__(self, nc=80, hidc=256, block_num=2, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.na = 3
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.conv = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
        self.dyhead = nn.Sequential(*[DyHeadBlock(hidc) for i in range(block_num)])
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(hidc, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(hidc, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
 
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
        x = self.dyhead(x)
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
 
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
 
   





    # stride = torch.empty(0)  # strides computed during build
    # export = False  # onnx export
    # end2end = False
    # include_nms = False
    # concat = False# init
    # shape = None
    # dynamic = False

 
    # def __init__(self, nc=80, anchors=(),ch=(),hidc=256, block_num=2):  # detection layer
    #     super().__init__()
    #     self.nc = nc  # number of classes
    #     self.no = nc + 5  # number of outputs per anchor
    #     self.nl = len(anchors)  # number of detection layers
    #     self.na = len(anchors[0]) // 2  # number of anchors
    #     self.grid = [torch.zeros(1)] * self.nl  # init grid
    #     a = torch.tensor(anchors).float().view(self.nl, -1, 2)
    #     self.register_buffer('anchors', a)  # shape(nl,na,2)
    #     self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1

    #     # self.nc = nc  # number of classes
    #     # self.nl = len(ch)  # number of detection layers
    #     self.reg_max = 1  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x
    #     self.stride = torch.zeros(self.nl)  # strides computed during build
    #     c2, c3 =  self.reg_max * 4,  self.nc  # channels
    #     self.conv = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
    #     self.dyhead = nn.Sequential(*[DyHeadBlock(hidc) for i in range(block_num)])
    #     self.cv2 = nn.ModuleList(
    #         nn.Sequential(Conv(hidc, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for _ in ch)
    #     self.cv3 = nn.ModuleList(nn.Sequential(Conv(hidc, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.no, 1)) for _ in ch)
    #     self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    #     self.CONV101 = nn.ModuleList(nn.Sequential(nn.Conv2d(4 * self.reg_max+self.no , self.no*self.nl,kernel_size=3,padding=1,stride=1)) for _ in range(len(ch)) )
 
    # def forward(self, x):
    #     """Concatenates and returns predicted bounding boxes and class probabilities."""
    #     z = []      
    #     for i in range(self.nl):
    #         x[i] = self.conv[i](x[i])
    #     x = self.dyhead(x)
    #     shape = x[0].shape  # BCHW
    #     for i in range(self.nl):
    #         x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
       
    #     for i in range(self.nl):
    #         x[i] = self.CONV101[i](x[i])

    #         h=x
        
       

    #         bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            
        
    #         x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    #     if self.training:
    #         return x
        
    #     elif self.dynamic or self.shape != shape:
    #         self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
    #         self.shape = shape

    #     x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        

    #     box, cls = x_cat.split((4, self.nc), 1)
    #     dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    #     y = torch.cat((dbox, cls.sigmoid()), 1)
    #     return (y,x) 

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2 , m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
 
       
    

    @staticmethod
    def _make_grid(nx=20, ny=20):
        
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

# if __name__ == '__main__':


#     shape = (1,128,16,16)  # 举例：创建一个形状为3x4x5的tensor
#     zeros_tensor1 = torch.zeros(shape)  
#     shape = (1,256,8,8)  # 举例：创建一个形状为3x4x5的tensor
#     zeros_tensor2 = torch.zeros(shape)    
#     shape = (1,512,4,4)  # 举例：创建一个形状为3x4x5的tensor
#     zeros_tensor3 = torch.zeros(shape)   
#     input = [zeros_tensor1,zeros_tensor2,zeros_tensor3]

#     ww = Detect_DyHead(nc=10,anchors=[[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]],ch=[128,256,512])
#     y=ww(input)
