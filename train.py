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
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--logging", help="Enable verbose mode", action="store_true")
parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")


args = parser.parse_args()

arg_alpha = args.alpha
arg_beta = args.beta
runname = args.runname
projectname = args.projectname
arg_logging = args.logging
arg_nottest = args.nottest


if arg_logging:
    wandb.init(project=projectname, entity="saraa_team", name=runname)

data_path = '/root/home/MD/'

train_images, train_masks = data_pred(data_path, 'train')
val_images, val_masks = data_pred(data_path, 'val')

train_dataset = DataPrep(train_images, train_masks, transform=transform)
val_dataset = DataPrep(val_images, val_masks, transform=transform)

BATCH_SIZE = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

model = Network(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
epochs = args.epochs

for epoch in range(0, epochs):
  
  loss_function = TverskyCrossEntropyLcDiceWeightedLoss(2, arg_alpha, arg_beta, 4/3, 0.8, 0.2)
  lrr = 1e-4
  
  optimizer = torch.optim.Adam(model.parameters(), lr=lrr, weight_decay=1e-3)

  total_train_loss = 0
  train_count = 0
    
  total_val_loss = 0
  val_count = 0
    
  total_val_miou = 0
  val_average = 0
  val_count = 0
  val_miou = 0    
  val_class_iou = 0
  total_val_class_iou = 0
    
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
    loss = loss_function(mask, train_y)
      
    total_train_loss += loss.item()

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
      val_loss = loss_function(mask, val_y)

    total_val_loss += val_loss.item()
      
    x = x.detach()
    mask = mask.detach()
      
    mask = torch.argmax(mask, dim=1).detach().cpu().numpy()  
    val_y = val_y.squeeze().detach().cpu().numpy()  
    mask = mask.squeeze(0)

    comm, corr, qual = relaxed_f1(mask, val_y, 3)
    tmiou, ciou = mIoU(mask, val_y, 2)

    val_miou += tmiou
    val_class_iou += ciou

    val_comm += comm
    val_corr += corr
    val_qual += qual

    total_val_comm += val_comm
    total_val_corr += val_corr
    total_val_qual += val_qual

    total_val_miou += val_miou
    val_miou = 0
    val_comm = 0
    val_corr = 0
    val_qual = 0
    val_count += 1
      
    if not(arg_nottest):
        break

  val_average = total_val_loss / val_count
  total_val_class_iou = val_class_iou / val_count

  val_comm_avg = total_val_comm / val_count
  val_corr_avg = total_val_corr / val_count
  val_qual_avg = total_val_qual / val_count

  print(f"Epoch: {epoch+1}, Training_loss: {train_average}, Validation_loss: {val_average}")
  print('\n', f"val_class_iou: {total_val_class_iou}, Val_mIoU: {val_average*100}")
  print('\n', f"val_comm_avg: {val_comm_avg}, val_corr_avg: {val_corr_avg}, val_qual_avg: {val_qual_avg}")
    
  if arg_logging:
      wandb.log({"Training Loss": train_average, "Validation Loss": val_average, "Val_mIoU": val_average, "val_comm_avg": val_comm_avg, "val_corr_avg": val_corr_avg, "val_qual_avg": val_qual_avg})
      os.makedirs('../saved_models', exist_ok=True)
      torch.save(model.state_dict(), f'../saved_models/SemSeg_combinedloss_a{arg_alpha}b{arg_beta}_epoch_{epoch+1}.pth')
      artifact = wandb.Artifact(f'SemSeg_combinedloss_a{arg_alpha}b{arg_beta}_epoch{epoch+1}', type='model')
      artifact.add_file(f'../saved_models/SemSeg_combinedloss_a{arg_alpha}b{arg_beta}_epoch{epoch+1}.pth')
      wandb.log_artifact(artifact)

