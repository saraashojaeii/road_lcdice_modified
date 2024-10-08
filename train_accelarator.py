from utils import *
from data import *
from Network import *
from losses import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb
import random
import argparse

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description="A script with argparse options")


parser.add_argument("--runname", type=str, required=False)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--projectname", type=str, required=False)


runname = args.runname
projectname = args.projectname
arg_dataset = args.dataset_name


logging = True

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])

current_rank = accelerator.state.process_index
num_gpus = accelerator.state.num_processes

if logging:
    if accelerator.is_main_process:
      wandb.init(project=projectname, entity="saraa_team", name=runname)

data_path = '/root/home/MD/'
# data_path = '/home/sara/Docker_file/massachusetts-roads-dataset/'

train_images, train_masks = data_pred(data_path, 'train', arg_dataset)
val_images, val_masks = data_pred(data_path, 'val', arg_dataset)

train_dataset = DataPrep(train_images, train_masks, transform=transform)
val_dataset = DataPrep(val_images, val_masks, transform=transform)

BATCH_SIZE = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

model = Network(1)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {num_params} trainable parameters.")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = accelerator.device

model.to(device)
epochs = 200

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
# gap_loss_fn = GapLoss(K=1)
# mse_loss_fn = nn.MSELoss()

criterion = BCE_SACone_lcDice(2, 1, 0.4, 4/3, 0.8, 0.2, 0)


arg_nottest = True
arg_logging = True


model, optimizer, training_dataloader = accelerator.prepare(
        model, optimizer, train_loader
    )
validation_dataloader = accelerator.prepare(val_loader)


for epoch in range(0, epochs):

  total_train_loss = 0
  total_val_loss = 0
  train_count = 0
  val_count = 0
  val_average = 0
    
  val_comm = 0
  val_corr = 0
  val_qual = 0
  total_val_comm = 0
  total_val_corr = 0
  total_val_qual = 0 

  # Training loop
  for sample in tqdm(training_dataloader):
    model.train()
      
    train_x, train_y = sample
    # train_x, train_y = train_x.to(device), train_y

    optimizer.zero_grad()
      
    mask, x = model(train_x)
    # gap_loss = gap_loss_fn(mask, train_y)
    # mse_loss = mse_loss_fn(mask, train_y)
    
    loss = criterion(mask, train_y)
      
    # loss.backward()
    accelerator.backward(loss)
    optimizer.step()

    total_train_loss += loss.item()
    train_count += 1
    if not(arg_nottest):
        break
  train_average = total_train_loss / train_count
  
  # Validation loop
  for sample in validation_dataloader:
    model.eval()
    val_x, val_y = sample
    val_x, val_y = val_x.to(device), val_y.to(device)

    with torch.no_grad():
      mask, x = model(val_x)
      # gap_loss = gap_loss_fn(mask, val_y)
      # mse_loss = mse_loss_fn(mask, val_y)

      val_loss = criterion(mask, val_y)

    total_val_loss += val_loss.item()
    val_count += 1
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

  val_average = total_val_loss / val_count

  if accelerator.is_main_process:
      
  
    if logging:
        wandb.log({"Epoch": (epoch+1), "Training Loss": train_average, "Validation Loss": val_average, "val_comm_avg": val_comm_avg, "val_corr_avg": val_corr_avg, "val_qual_avg": val_qual_avg, "val_f1": val_f1})
        os.makedirs('../saved_models', exist_ok=True)
        torch.save(model.state_dict(), f'../saved_models/SemSeg_coneLoss_epoch{epoch+1}.pth')
        artifact = wandb.Artifact(f'SemSeg_coneLoss_epoch{epoch+1}', type='model')
        artifact.add_file(f'../saved_models/SemSeg_coneLoss_epoch{epoch+1}.pth')
        wandb.log_artifact(artifact)

  # print(f"Epoch: {epoch+1}, Training_loss: {train_average}, Validation_loss: {val_average}")


  # if epoch%10==0:
  # torch.save(model.state_dict(),f"./SemSeg_with_gaplossL2_ep{(epoch+1)}.pth")
  # print('          MODEL SAVED          ')

wandb.finish()
