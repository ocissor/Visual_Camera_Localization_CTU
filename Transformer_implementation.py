#Transformer model implementation
import math
from typing import Tuple
from tqdm import tqdm
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import os
import pickle
import sys
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.insert(1,'/home/batrasar/code/relpose_gnn/python')
from niantic.utils.pose_utils import quaternion_angular_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerVL(nn.Module):
  def __init__(self,d_model,nhead):
    super().__init__()
    self.encoder = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead)

  def forward(self, src):
    out = self.encoder(src)
    return out


class DatasetVal(torch.utils.data.Dataset):
  def __init__(self, args):
    super().__init__()
    self.b_pth = args.baseline_data_dir
    self.test_scene = args.test_scene
    self.seq_len = args.seqlen
    self.scenes_info_path = os.path.join(self.b_pth,'K_'+str(self.seq_len-1))
    q_poses_pth = os.path.join(self.scenes_info_path,self.test_scene+'_q_poses.pkl')
    nn_poses_pth = os.path.join(self.scenes_info_path,self.test_scene+'_nn_poses.pkl')
    q_fv_pth = os.path.join(self.scenes_info_path,self.test_scene+'_q_fv.pkl')
    nn_fv_pth = os.path.join(self.scenes_info_path,self.test_scene+'_nn_fv.pkl')

    with open(q_poses_pth,'rb') as f:
      self.q_poses = pickle.load(f)

    with open(nn_poses_pth,'rb') as f:
      self.nn_poses = pickle.load(f)

    with open(q_fv_pth,'rb') as f:
      self.q_fv = pickle.load(f)

    with open(nn_fv_pth,'rb') as f:
      self.nn_fv = pickle.load(f)


  def __getitem__(self, index):
    label = torch.from_numpy(self.q_poses[index])
    len_gt = len(self.q_poses[index])
    input = torch.zeros(self.seq_len,len_gt)#(8,7)
    input[0] = torch.from_numpy(self.q_poses[index])
    for i in range(1,self.seq_len):
      input[i] = torch.from_numpy(self.nn_poses[index][i-1])

    return (input,label)

  def __len__(self):
    return len(self.q_poses)

class DatasetTrain(torch.utils.data.Dataset):
  def __init__(self, args):
    super().__init__()
    self.b_pth = args.baseline_data_dir
    self.test_scene = args.test_scene
    self.seq_len = args.seqlen
    self.scenes_info_path = os.path.join(self.b_pth,'K_'+str(self.seq_len-1))
    q_poses_pth = os.path.join(self.scenes_info_path,self.test_scene+'_q_poses_train.pkl')
    nn_poses_pth = os.path.join(self.scenes_info_path,self.test_scene+'_nn_poses_train.pkl')
    q_fv_pth = os.path.join(self.scenes_info_path,self.test_scene+'_q_fv_train.pkl')
    nn_fv_pth = os.path.join(self.scenes_info_path,self.test_scene+'_nn_fv_train.pkl')

    with open(q_poses_pth,'rb') as f:
      self.q_poses_train = pickle.load(f)

    with open(nn_poses_pth,'rb') as f:
      self.nn_poses_train = pickle.load(f)

    with open(q_fv_pth,'rb') as f:
      self.q_fv_train = pickle.load(f)

    with open(nn_fv_pth,'rb') as f:
      self.nn_fv_train = pickle.load(f)


  def __getitem__(self, index):
    label = torch.from_numpy(self.q_poses_train[index])
    len_gt = len(self.q_poses_train[index])
    input = torch.zeros(self.seq_len,len_gt)#(8,7)
    input[0] = torch.from_numpy(self.q_poses_train[index])
    for i in range(1,self.seq_len):
      input[i] = torch.from_numpy(self.nn_poses_train[index][i-1])

    return (input,label)

  def __len__(self):
    return len(self.q_poses_train)

class TransformerCriterionTrain(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0,learn_beta = True,saq=0):
        super().__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.tensor([sax],dtype=torch.float,device=device), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.tensor([saq],dtype=torch.float,device=device), requires_grad=learn_beta)

    def forward(self, pred, targ):
        """
        :param pred: N x 7
        :param targ: N x 7
        :return:
        """
        t_loss = self.t_loss_fn(pred[:, :3], targ[:, :3])
        q_loss = self.q_loss_fn(pred[:, 3:], targ[:, 3:])
        sax_exp = torch.exp(-self.sax)
        saq_exp = torch.exp(-self.saq)
        sax_exp = sax_exp.to(device)
        saq_exp = saq_exp.to(device)
        t_loss = t_loss.to(device)
        q_loss = q_loss.to(device)
        loss = sax_exp * t_loss + \
               self.sax + \
               saq_exp * q_loss + \
               self.saq
        return loss


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline-data-dir', type=str, help='Path to data for baseline evaluation',
                        default='/mnt/data-7scenes-ozgur/3dv/data/seven_scenes/')
  parser.add_argument('--test-scene', type=str, help='Which scene to test on', default='chess')
  parser.add_argument('--seqlen',type=int, help='length of sequence', default=8)
  parser.add_argument('--gpu', default=None, help='GpuId', type=int)
  args = parser.parse_args()
  #print(args.test_data_dir)

  print("Current scene is {}".format(args.test_scene))
  batch_size = 3
  learning_rate = 0.0001
  train_dataset = DatasetTrain(args)
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=1)
  val_dataset = DatasetVal(args)
  val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1, shuffle=True,
                                             num_workers=1)
  #(3,8,7)
  #label will be of shape (3,7)
  model = TransformerVL(7,7)#d_model = 7 and n_heads = 7 as d_model should be divisible by n_heads
  #src = torch.ones(3,8,7) means batch_size= 3 number of datapoints per input = 8(its equal to seqlen) and feature vector per datapoint = 7
  #model.forward(src)
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  #loss_fn = nn.MSELoss()
  loss_fn = TransformerCriterionTrain()
  val_t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
  val_q_criterion = quaternion_angular_error
  epochs = 50
  learning_rate = 0.0001

  for epoch in tqdm(range(epochs)):
    print('#######################')
    print('Epoch number {}'.format(epoch))


    train_batch_losses = []
    val_trans_loss = []
    val_rot_loss = []
    model.train()
    for i,data in enumerate(train_loader):
      input,label = data
      label = Variable(label)
      label = label.to(device)
      input = Variable(input)
      input = input.to(device)
      optimizer.zero_grad()
      out = model.forward(input.float())
      out = out[:,0,:]
      loss = loss_fn.forward(out,label)
      loss.backward()#we calculate the gradient of parameters w.r.t loss 
      train_batch_losses.append(loss.data.cpu().numpy())
      optimizer.step()#update he parameters

    print("Median Training Loss for epoch {} is {}".format(epoch,np.median(train_batch_losses)))

    model.eval()
    for val_data in val_loader:
      with torch.no_grad():
        val_input,val_label = val_data
        val_label = Variable(val_label)
        val_label = val_label.to(device)
        val_input = Variable(val_input)
        val_input = val_input.to(device)
        val_out = model.forward(val_input)
        val_out = val_out[:,0,:]
        pred_rot = np.asarray(val_out[:,3:].squeeze(0).detach().cpu())
        pred_rot = pred_rot/(np.linalg.norm(pred_rot))
        tar_rot = np.asarray(val_label[:,3:].squeeze(0).detach().cpu())
        trans_loss = val_t_criterion(np.asarray(val_out[:,:3].detach().cpu()),np.asarray(val_label[:,:3].detach().cpu()))
        rot_loss = val_q_criterion(pred_rot, tar_rot)
        val_trans_loss.append(trans_loss)
        val_rot_loss.append(rot_loss)

    median_error_trans = np.median(np.asarray(val_trans_loss))
    median_error_rot = np.median(np.asarray(val_rot_loss))
    print("median rot error is {} for scene {}".format(median_error_rot,args.test_scene))
    print("median trans error is {} for scene {}".format(median_error_trans,args.test_scene))

    print("##################")
    

################################################################################################################






 




