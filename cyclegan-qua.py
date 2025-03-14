import os
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import itertools
from absl import flags, app
from torchvision import transforms
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tensorboardX import SummaryWriter
from tqdm import tqdm
import source.models.dcgan as models
import source.losses as losses
from source.utils import set_seed

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
##################################################
#  CycleGAN
##################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x):
        return x + self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self, input_nc=1, z_dim=256, n_residual_blocks=9):
        super(Encoder, self).__init__()
        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
class Decoder(nn.Module):
    def __init__(self, z_dim=256, output_nc=1):
        super(Decoder, self).__init__()
        # Upsampling
        model = []
        in_features = z_dim
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]
        self.model = nn.Sequential(*model)
    def forward(self, z):
        return self.model(z)
    
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        x =  self.model(x)
        return x

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

###################
# dataloader
###################

# 图像大小256X256
size = 256
# resize到256X256大小，减少图像大小，只在X-Y平面做
def resize_image_itk(ori_img, size=256, resamplemethod=sitk.sitkNearestNeighbor):
    original_spacing = ori_img.GetSpacing()
    original_size = ori_img.GetSize()
    out_spacing = [
                    original_size[0]*original_spacing[0] / size,
                    original_size[1]*original_spacing[1] / size,
                    original_spacing[2]] # 
    out_size = [size, size, original_size[-1]] # [480,480,ori]
    #print(out_spacing)
    #print(out_size)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(ori_img.GetDirection())
    resample.SetOutputOrigin(ori_img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(ori_img.GetPixelIDValue())
    resample.SetInterpolator(resamplemethod)
    if resamplemethod == sitk.sitkNearestNeighbor: # 如果是mask图像，就选择sitkNearestNeighbor这种插值
        resample.SetOutputPixelType(sitk.sitkInt16)
    else: # 如果是普通图像，就采用sitkLiner插值法
        resample.SetOutputPixelType(sitk.sitkFloat32)

    return resample.Execute(ori_img) 
# resize + 剔除异常值
def readimage(path):
    files = os.listdir(path)
    files = sorted(files, key=lambda x: int(x.split("_")[0][7:]))
    fileroot = [os.path.join(path,name) for name in files]
    slices = []
    for i in fileroot:
        # resize
        data = niiloader(i)
        # 整体上剔除异常值
        outliers = remove_outliers(data=data)
        slices.append(outliers)
        #print(outliers.shape)
    return np.concatenate(slices,axis=0)
def readmask(path):
    files = os.listdir(path)
    files = sorted(files, key=lambda x: int(x.split("_")[0][7:]))
    fileroot = [os.path.join(path,name) for name in files]
    slices = []
    for i in fileroot:
        data = niiloader(i,label=True)
        slices.append(data)
    return np.concatenate(slices,axis=0)
def niiloader(file,label=False,resize=True):
    inputType = sitk.sitkInt16 if label else sitk.sitkFloat32
    resamplemethod = sitk.sitkNearestNeighbor if label else sitk.sitkLinear
    vol = sitk.ReadImage(file,outputPixelType=inputType)
    if resize:
        vol = resize_image_itk(vol,size=size,resamplemethod=resamplemethod)
    data = sitk.GetArrayFromImage(vol)
    return data
def minmax(input):
    scaler = (input - torch.min(input))/(torch.max(input) - torch.min(input)) # convert to [0,1]
    normal = (scaler - 0.5) / 0.5 # convert to [-1,1]
    return normal
# 去除极值，保存百分位0到99.5%的值，大于这个的值应该是异常点，反映在lge对比度太低了
def remove_outliers(data, lower_percentile=0, upper_percentile=99.5):
    # 计算百分位数的阈值
    lower_threshold = np.percentile(data, lower_percentile)
    upper_threshold = np.percentile(data, upper_percentile)
    #print(lower_threshold,upper_threshold)
    # 使用布尔索引将小于下界和大于上界的值设置为0
    filtered_data = data[(data >= lower_threshold) & (data <= upper_threshold)]
    Min = np.min(filtered_data)
    Max = np.max(filtered_data)
    data[data<lower_threshold] = Min
    data[data>upper_threshold] = Max
    return data
# 裁剪到256X256大小
def central_crop(array:np,size):
    # array大于size
    if array.shape[-1] > size or array.shape[-2] > size:
        # 计算裁剪起始和结束为止
        start_row = (array.shape[-2] - size) // 2
        end_row = start_row + size
        start_col = (array.shape[-1] - size) // 2
        end_col = start_col + size
        if array.ndim == 3:
            cropped_array = array[:, start_row:end_row, start_col:end_col]
        elif array.ndim == 2:
            cropped_array = array[start_row:end_row, start_col:end_col]
        else:
            print("警告:不是2维或3维数组")
            exit()
        return cropped_array
    
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    
trans = transforms.RandomHorizontalFlip()
class ImageDataset(Dataset):
    '''
    Dataloader of MS-Seg 2019
    '''
    def __init__(self, domain_a, domain_b, root):
        self.img_A = readimage(os.path.join(root,domain_a))
        self.img_B = readimage(os.path.join(root,domain_b))
        self.len_a = len(self.img_A)
        self.len_b = len(self.img_B)
        print(self.len_a)
        print(self.len_b)

    def __getitem__(self, index):
        index_a = index % self.len_a
        index_b = index % self.len_b 
        # apply torch transform
        a = self.img_A[index_a] # H X W
        a_tensor = torch.from_numpy(a)
        item_A = trans(a_tensor)
        item_A = minmax(item_A)
        # random load B domain image
        b = self.img_B[random.randint(0, self.len_b - 1)]# H X W
        b_tensor = torch.from_numpy(b)
        item_B = trans(b_tensor)
        item_B = minmax(item_B)
        
        return {'A':torch.unsqueeze(item_A,dim=0), 'B':torch.unsqueeze(item_B,dim=0)}
    def __len__(self):
        return max(self.len_a, self.len_b)

identity_loss = torch.nn.L1Loss()
cycleg_loss = torch.nn.L1Loss()
net_G_models = {
    'cnn32': models.Generator32,
    'cnn48': models.Generator48,
}

net_D_models = {
    'cnn32': models.Discriminator32,
    'cnn48': models.Discriminator48,
}

loss_fns = {
    'bce': losses.BCEWithLogits,
    'mse': losses.MSE,
    'hinge': losses.Hinge,
    'was': losses.Wasserstein,
    'softplus': losses.Softplus
}

FLAGS = flags.FLAGS
# model and training
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'stl10', 'lsun', 'celeba', 'imagenet'], "dataset")
flags.DEFINE_enum('arch', 'cnn32', net_G_models.keys(), "architecture")
flags.DEFINE_integer('total_steps', 50000, "total number of training steps")
flags.DEFINE_integer('batch_size', 1, "batch size")
flags.DEFINE_float('lr_G', 2e-4, "Generator learning rate")
flags.DEFINE_float('lr_D', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.5, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 1, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 100, "latent space dimension")
flags.DEFINE_enum('loss', 'mse', loss_fns.keys(), "loss function")
flags.DEFINE_integer('seed', 2024, "random seed")
# logging
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('eval_epoch', 10, "save model per eval_epoch")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/cyclegan-qua', 'logging folder')
flags.DEFINE_bool('record', True, "record inception score and FID score")
flags.DEFINE_string('fid_cache', './stats/cifar10_stats.npz', 'FID cache')
# generate
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_bool('sample_shift', True, 'sample_shift')
flags.DEFINE_string('pretrain', './logs/cyclegan-qua/models/model-epoch-99.pt', 'path to test model')
flags.DEFINE_string('output', './outputs', 'path to output dir')
flags.DEFINE_integer('num_images', 50000, 'the number of generated images')
# advlatgan parameters
flags.DEFINE_integer('num_iter', 1, "number of adversarial iterations")
flags.DEFINE_integer('sf_iter', 3, "number of sample shift")
flags.DEFINE_float('eps', 0.01, "epsilon of I-FGSM")
flags.DEFINE_float('lambda_B', 10.0, "intensity of D_B loss")
flags.DEFINE_float('lambda_A', 10.0, "intensity of D_A loss")
flags.DEFINE_float('lambda_idt', 10.0, "intensity of identity loss")

device = torch.device('cuda:0')
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

def infiniteloop(dataloader):
    while True:
        for x, batch in iter(dataloader):
            yield batch
            
# sample shift
def sample_shift():
    assert FLAGS.pretrain is not None, "set model weight by --pretrain [model]"
    # domain A 2 domain B
    a2b_encoder= Encoder().to(device)
    a2b_decoder= Decoder().to(device)
    dis_a2b = Discriminator(1).to(device)
    a2b_encoder.load_state_dict(torch.load(FLAGS.pretrain)['net_a2b_encoder'])
    a2b_decoder.load_state_dict(torch.load(FLAGS.pretrain)['net_a2b_decoder'])
    dis_a2b.load_state_dict(torch.load(FLAGS.pretrain)['dis_a2b'])
    a2b_encoder.eval()
    a2b_decoder.eval()
    transimageroot = "/home/hfcui/disk1/ms-seg-2019/train/c0"
    transmaskroot = "/home/hfcui/disk1/ms-seg-2019/train/c0gt"
    imagefile = os.listdir(transimageroot)
    imagefile = sorted(imagefile, key=lambda x: int(x.split("_")[0][7:]))
    maskfile = os.listdir(transmaskroot)
    maskfile = sorted(maskfile, key=lambda x: int(x.split("_")[0][7:]))
    loss_fn = loss_fns[FLAGS.loss]()

    for i in range(len(maskfile)):
        imageroot = os.path.join(transimageroot,imagefile[i])
        maskroot = os.path.join(transmaskroot,maskfile[i])
        imgvol = sitk.ReadImage(imageroot,sitk.sitkFloat32)
        maskvol = sitk.ReadImage(maskroot,sitk.sitkInt16)
        imgres = resize_image_itk(imgvol,resamplemethod=sitk.sitkLinear)
        maskres = resize_image_itk(maskvol,resamplemethod=sitk.sitkNearestNeighbor)
        # img must equal to mask
        direction = imgres.GetDirection()
        spacing = imgres.GetSpacing()
        origin = imgres.GetOrigin()
        data = sitk.GetArrayFromImage(imgres)
        data = remove_outliers(data)

        out = np.zeros_like(data)
        for j in range(data.shape[0]):
            input = torch.from_numpy(data[j]).unsqueeze(dim=0) # 1xhxw
            input = minmax(input).unsqueeze(dim=0).to(device)
            # do sample_shift
            z = a2b_encoder(input)
            z_adv = Variable(z.data, requires_grad=True).to(device)
            for k in range(FLAGS.sf_iter):
                a2b_decoder.zero_grad()
                a2b_encoder.zero_grad()
                dis_a2b.zero_grad()
                if z_adv.grad is not None:
                    z_adv.grad.data.fill_(0)
                loss = loss_fn(dis_a2b(a2b_decoder(z_adv)))
                loss.backward()
                z_adv.grad.sign_()
                z_adv = z_adv - z_adv.grad * FLAGS.eps
                z_adv = Variable(z_adv.data, requires_grad=True).to(device)
            
            fake_lge = a2b_decoder(z_adv)
            output = fake_lge.detach().cpu().numpy().squeeze() # [h,w]
            out[j] = output
        out = sitk.GetImageFromArray(out)
        out.SetDirection(direction)
        out.SetSpacing(spacing)
        out.SetOrigin(origin)
        sitk.WriteImage(out,'/home/hfcui/disk1/ms-seg-2019/train/advcyc_z_c0_lge_e100/'+imagefile[i])
        #sitk.WriteImage(maskres,'/home/hfcui/disk1/ms-seg-2019/train/c0_lge_gt/'+maskfile[i])
        #sitk.WriteImage(imgres,'/home/hfcui/disk1/ms-seg-2019/train/c0_res/'+imagefile[i])


def generate():
    assert FLAGS.pretrain is not None, "set model weight by --pretrain [model]"
    # domain A 2 domain B
    a2b_encoder= Encoder().to(device)
    a2b_decoder= Decoder().to(device)
    a2b_encoder.load_state_dict(torch.load(FLAGS.pretrain)['net_a2b_encoder'])
    a2b_decoder.load_state_dict(torch.load(FLAGS.pretrain)['net_a2b_decoder'])
    a2b_encoder.eval()
    a2b_decoder.eval()

    #root = '/home/hfcui/disk1/ms-seg-2019/train/c0'
    transimageroot = "/home/hfcui/disk1/ms-seg-2019/train/c0"
    transmaskroot = "/home/hfcui/disk1/ms-seg-2019/train/c0gt"
    imagefile = os.listdir(transimageroot)
    imagefile = sorted(imagefile, key=lambda x: int(x.split("_")[0][7:]))
    maskfile = os.listdir(transmaskroot)
    maskfile = sorted(maskfile, key=lambda x: int(x.split("_")[0][7:]))

    with torch.no_grad():
        for i in range(len(maskfile)):
            imageroot = os.path.join(transimageroot,imagefile[i])
            maskroot = os.path.join(transmaskroot,maskfile[i])
            imgvol = sitk.ReadImage(imageroot,sitk.sitkFloat32)
            maskvol = sitk.ReadImage(maskroot,sitk.sitkInt16)
            imgres = resize_image_itk(imgvol,resamplemethod=sitk.sitkLinear)
            maskres = resize_image_itk(maskvol,resamplemethod=sitk.sitkNearestNeighbor)
            # img must equal to mask
            direction = imgres.GetDirection()
            spacing = imgres.GetSpacing()
            origin = imgres.GetOrigin()
            data = sitk.GetArrayFromImage(imgres)
            data = remove_outliers(data)
    
            out = np.zeros_like(data)
            for j in range(data.shape[0]):
                input = torch.from_numpy(data[j]).unsqueeze(dim=0) # 1xhxw
                input = minmax(input).unsqueeze(dim=0).to(device)
                fake_lge = a2b_decoder(a2b_encoder(input))
                output = fake_lge.detach().cpu().numpy().squeeze() # [h,w]
                out[j] = output
            out = sitk.GetImageFromArray(out)
            out.SetDirection(direction)
            out.SetSpacing(spacing)
            out.SetOrigin(origin)
            sitk.WriteImage(out,'/home/hfcui/disk1/ms-seg-2019/train/advcyc_c0_lge_e200/'+imagefile[i])
            #sitk.WriteImage(maskres,'/home/hfcui/disk1/ms-seg-2019/train/c0_lge_gt/'+maskfile[i])
            #sitk.WriteImage(imgres,'/home/hfcui/disk1/ms-seg-2019/train/c0_res/'+imagefile[i])


def train():
    # 加载数据集
    dataset = ImageDataset(domain_a='c0',domain_b='lge',root='/home/hfcui/disk1/ms-seg-2019/train')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    # 定义网络生成网络
    a2b_encoder = Encoder().to(device)
    a2b_decoder = Decoder().to(device)
    b2a_encoder = Encoder().to(device)
    b2a_decoder = Decoder().to(device)
    # 定义判别网络
    D_a2b = Discriminator(1).to(device)
    D_b2a = Discriminator(1).to(device)
    # 定义损失函数
    loss_fn = loss_fns[FLAGS.loss]()
    # 定义优化器
    optim_G = optim.Adam(itertools.chain(a2b_encoder.parameters(),a2b_decoder.parameters(),
                                        b2a_encoder.parameters(),b2a_decoder.parameters() ), lr=FLAGS.lr_G, betas=FLAGS.betas)
    optim_D = optim.Adam(itertools.chain(D_a2b.parameters(),D_b2a.parameters()), lr=FLAGS.lr_D, betas=FLAGS.betas)
    # 定义学习率下降策略
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=LambdaLR(200, 0 ,100).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optim_D, lr_lambda=LambdaLR(200, 0 ,100).step)
    # 创建输出文件夹
    os.makedirs(os.path.join(FLAGS.logdir, 'origin-A'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.logdir, 'origin-B'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.logdir, 'translate-B'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.logdir, 'translate-A'), exist_ok=True)
    
    os.makedirs(os.path.join(FLAGS.logdir, 'models'), exist_ok=True)
    writer = SummaryWriter(os.path.join(FLAGS.logdir))
    #sample_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
    
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    writer.add_text(
        "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))
    
    # 设置epoch
    total_epochs = 200
    print(len(dataloader))
    for epoch in range(total_epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader),ncols=140)
        for index,batch in loop:
            step = epoch * len(dataloader) + index
            loop.set_description(f'Epoch [{epoch}/{total_epochs}]')
            # 真实数据 BX1XHXW
            reala = batch['A'].to(device)
            realb = batch['B'].to(device)
            ###############################################
            # 从生成器的编码器产生潜在空间编码
            za = a2b_encoder(reala)
            zb = b2a_encoder(reala)
            
            za_adv = Variable(za.data, requires_grad=True).to(device)
            zb_adv = Variable(zb.data, requires_grad=True).to(device)
            
            for j in range(FLAGS.num_iter):
                optim_D.zero_grad()
                optim_G.zero_grad()
                if za_adv.grad is not None:
                    za_adv.grad.data.fill_(0)
                if zb_adv.grad is not None:
                    zb_adv.grad.data.fill_(0)    
                lossa = loss_fn(D_a2b(a2b_decoder(za_adv)))
                lossb = loss_fn(D_b2a(b2a_decoder(zb_adv)))
                lossa.backward()
                lossb.backward()
                za_adv.grad.sign_()
                zb_adv.grad.sign_()
                za_adv = za_adv - za_adv.grad * FLAGS.eps
                zb_adv = zb_adv - zb_adv.grad * FLAGS.eps
                za_adv = Variable(za_adv.data, requires_grad=True).to(device)
                zb_adv = Variable(zb_adv.data, requires_grad=True).to(device)
            za = za_adv
            zb = zb_adv
            #  Discriminator A & B
            optim_D.zero_grad()
            with torch.no_grad():
                fakeb = a2b_decoder(za).detach()        
            # '''不加入buffer'''
            net_Db_real = D_a2b(realb)
            fake_B_pool = fake_B_buffer.push_and_pop(fakeb)
            net_Db_fake = D_a2b(fake_B_pool.detach())
            #net_Db_fake = D_a2b(fakeb)
            lossDb = loss_fn(net_Db_real, net_Db_fake)
            lossDb.backward()
            with torch.no_grad():
                fakea = b2a_decoder(zb).detach()        
            # '''不加入buffer'''
            net_Da_real = D_b2a(reala)
            fake_A_pool = fake_A_buffer.push_and_pop(fakea)
            net_Da_fake = D_b2a(fake_A_pool.detach())
            #net_Da_fake = D_b2a(fakea)
            lossDa = loss_fn(net_Da_real, net_Da_fake)
            lossDa.backward()  
            optim_D.step()
            writer.add_scalar('DisB_loss', lossDb, step)
            writer.add_scalar('DisA_loss', lossDa, step)
            ##############################################

            #############################################
            # 训练生成器网络GA和GB
            #set_requires_grad([D_a2b,D_b2a],False)
            fake_B = a2b_decoder(a2b_encoder(reala))
            fake_A = b2a_decoder(b2a_encoder(realb))
            recon_A = b2a_decoder(b2a_encoder(fake_B))
            recon_B = a2b_decoder(a2b_encoder(fake_A))
            same_B = a2b_decoder(a2b_encoder(realb))
            same_A = b2a_decoder(b2a_encoder(reala))
            
            optim_G.zero_grad()
            # identity loss
            idt_B = identity_loss(realb,same_B)*FLAGS.lambda_B*FLAGS.lambda_idt
            idt_A = identity_loss(reala,same_A)*FLAGS.lambda_A*FLAGS.lambda_idt
            # gan loss
            lossGb = loss_fn(D_a2b(fake_B))
            lossGa = loss_fn(D_b2a(fake_A))
            # cyclegan loss
            lossCyca = cycleg_loss(reala,recon_A)*FLAGS.lambda_A
            lossCycb = cycleg_loss(realb,recon_B)*FLAGS.lambda_B
            # total loss
            lossT = idt_B + idt_A + lossGb + lossGa + lossCyca + lossCycb
            writer.add_scalar('Gena2b_loss', lossGb, step)
            writer.add_scalar('Genb2a_loss', lossGa, step)
            writer.add_scalar('Idta2a_loss', idt_A, step)
            writer.add_scalar('Idtb2b_loss', idt_B, step)
            writer.add_scalar('Cyca2a_loss', lossCyca, step)
            writer.add_scalar('Cycb2b_loss', lossCycb, step)
            lossT.backward()
            optim_G.step()
            ###############################################

            loop.set_postfix(Db='%.5f' % lossDb,
                             Da='%.5f' % lossDa,
                             Ga2b='%.5f' % lossGb,
                             Gb2a='%.5f' % lossGa,
                             #Ia2a='%.3f' % idt_A,
                             #Ib2b='%.3f' % idt_B,
                             #Ca2a='%.3f' % lossCyca,
                             Cb2b='%.3f' % lossCycb) 
            ###################################################
            if step == 1 or step % len(dataloader) == 0:
                TransB = (np.squeeze(fake_B.detach().cpu().numpy())+1) * 0.5
                TransA = (np.squeeze(fake_A.detach().cpu().numpy())+1) * 0.5
                A = (np.squeeze(reala.detach().cpu().numpy())+1) * 0.5
                B = (np.squeeze(realb.detach().cpu().numpy())+1) * 0.5
                #plt.imshow(A,)
                plt.imsave(os.path.join(
                    FLAGS.logdir, 'origin-A', '%d.png' % (epoch+1)),A,cmap='gray')
                plt.imsave(os.path.join(
                    FLAGS.logdir, 'origin-B', '%d.png' % (epoch+1)),B,cmap='gray')
                plt.imsave(os.path.join(
                    FLAGS.logdir, 'translate-B', '%d.png' % (epoch+1)),TransB,cmap='gray')
                plt.imsave(os.path.join(
                    FLAGS.logdir, 'translate-A', '%d.png' % (epoch+1)),TransA,cmap='gray')
                
        if (epoch+1) == 1 or (epoch+1) % FLAGS.eval_epoch == 0:
            torch.save({
                'net_a2b_encoder': a2b_encoder.state_dict(),
                'net_a2b_decoder': a2b_decoder.state_dict(),
                'net_b2a_encoder': b2a_encoder.state_dict(),
                'net_b2a_decoder': b2a_decoder.state_dict(),
                'dis_a2b': D_a2b.state_dict(),
                'dis_b2a': D_b2a.state_dict(),
                'optim_G': optim_G.state_dict(),
                'optim_D': optim_D.state_dict(),
                'sched_G': lr_scheduler_G.state_dict(),
                'sched_D': lr_scheduler_D.state_dict(),
            }, os.path.join(FLAGS.logdir, 'models', 'model-epoch-{:d}.pt'.format(epoch)))
            
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    elif FLAGS.sample_shift:
        sample_shift()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
