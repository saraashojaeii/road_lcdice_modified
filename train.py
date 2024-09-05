from utils import *
from data import *
from Network import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import random
import argparse

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description="A script with argparse options")

# Add an argument for an integer option
parser.add_argument("--runname", type=str, required=False)
parser.add_argument("--projectname", type=str, required=False)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--loss", type=str, required=False)
parser.add_argument("--k1", type=float, default=0.5, required=False)
parser.add_argument("--k2", type=float, default=0.5, required=False)
parser.add_argument("--k3", type=float, default=0.5, required=False)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--logging", help="Enable verbose mode", action="store_true")
parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")


args = parser.parse_args()
arg_loss = args.loss
arg_k1 = args.k1
arg_k2 = args.k2
arg_k3 = args.k3
runname = args.runname
projectname = args.projectname
arg_logging = args.logging
arg_nottest = args.nottest
arg_dataset = args.dataset_name

if arg_logging:
    wandb.init(project=projectname, entity="saraa_team", name=runname)

data_path = '/root/home/MD/'

train_images, train_masks = data_pred(data_path, 'train', arg_dataset)
val_images, val_masks = data_pred(data_path, 'val', arg_dataset)

train_dataset = DataPrep(train_images, train_masks, transform=transform)
val_dataset = DataPrep(val_images, val_masks, transform=transform)

BATCH_SIZE = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

model = Network(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
epochs = args.epochs

gap_loss_fn = GapLoss(K=1)
mse_loss_fn = nn.MSELoss()

for epoch in range(0, epochs):

  # if arg_loss == 'BCE_Tversky':
  #     loss_function = BCE_Tversky(2, 1, 0.4, 4/3, arg_k1, arg_k2)
  # if arg_loss == 'BCE_simpSAC':
  #     loss_function = BCE_simpSAC(2, 1, 0.4, 4/3, arg_k1, arg_k2)
  # if arg_loss == 'BCE_SimpSAC_lcDice':
  #     loss_function = BCE_SimpSAC_lcDice(2, 1, 0.4, 4/3, arg_k1, arg_k2, arg_k3)
  # if arg_loss == 'lcdice':
  #     loss_function = LcDiceLoss()
  # if arg_loss == 'BCE_SAC_lcDice':
  #    loss_function = BCE_SAC_lcDice(2, 1, 0.4, 4/3, arg_k1, arg_k2, arg_k3)
  
  lrr = 1e-4
  
  optimizer = torch.optim.Adam(model.parameters(), lr=lrr, weight_decay=1e-3)

  total_train_loss = 0
  train_count = 0
    
  total_val_loss = 0
  val_count = 0
  val_average = 0
    
  val_comm = 0
  val_corr = 0
  val_qual = 0
  total_val_comm = 0
  total_val_corr = 0
  total_val_qual = 0    

  for sample in tqdm(train_loader):
    model.train()
      
    train_x, train_y = sample
    train_x, train_y = train_x.to(device), train_y.to(device)

    optimizer.zero_grad()
      
    mask, x = model(train_x)
    # loss = loss_function(mask, train_y)
    # print(loss)
    # print(loss.shape)
    gap_loss = gap_loss_fn(mask, train_y)
    mse_loss = mse_loss_fn(mask, train_y)
    
    loss = gap_loss + mse_loss

    loss.backward()
    optimizer.step()

    total_train_loss += loss.item()
    train_count += 1
    if not(arg_nottest):
        break

  train_average = total_train_loss / train_count
  
  for sample in val_loader:
    model.eval()
    val_x, val_y = sample
    val_x, val_y = val_x.to(device), val_y.to(device)

    with torch.no_grad():
      mask, x = model(val_x)
      # val_loss = loss_function(mask, val_y)
    
      gap_loss = gap_loss_fn(mask, val_y)
      mse_loss = mse_loss_fn(mask, val_y)
      val_loss = gap_loss + mse_loss
      
    val_count += 1
    total_val_loss += val_loss.item()
      
    x = x.detach()
    mask = mask.detach()
      
    mask = torch.argmax(mask, dim=1).detach().cpu().numpy()  
    val_y = val_y.squeeze().detach().cpu().numpy()  
    mask = mask.squeeze(0)

    comm, corr, qual = relaxed_f1(mask, val_y, 3)
    tmiou, ciou = mIoU(mask, val_y, 2)

    val_comm += comm
    val_corr += corr
    val_qual += qual

    total_val_comm += val_comm
    total_val_corr += val_corr
    total_val_qual += val_qual

    val_comm = 0
    val_corr = 0
    val_qual = 0
      
    if not(arg_nottest):
        break

  val_average = total_val_loss / val_count

  val_comm_avg = total_val_comm / val_count
  val_corr_avg = total_val_corr / val_count
  val_qual_avg = total_val_qual / val_count
  val_f1 = 2 * (val_comm_avg * val_corr_avg)/(val_comm_avg + val_corr_avg)
  
  if arg_logging:
      wandb.log({"Epoch": (epoch+1), "Training Loss": train_average, "Validation Loss": val_average, "val_comm_avg": val_comm_avg, "val_corr_avg": val_corr_avg, "val_qual_avg": val_qual_avg, "val_f1": val_f1})
      os.makedirs('../saved_models', exist_ok=True)
      torch.save(model.state_dict(), f'../saved_models/SemSeg_combinedloss_epoch{epoch+1}.pth')
      artifact = wandb.Artifact(f'SemSeg_combinedloss_epoch{epoch+1}', type='model')
      artifact.add_file(f'../saved_models/SemSeg_combinedloss_epoch{epoch+1}.pth')
      wandb.log_artifact(artifact)

