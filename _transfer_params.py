'''
  Not intended to be included in git repo. For use on Broad cluster only.
'''
import os
import pandas as pd
import subprocess

def get_epoch_from_log_fn(log_fn):
  with open(log_fn) as f:
    lines = f.readlines()
  model_hyperparameters = lines[1].replace('Hyperparameters: ', '')
  x_dim = lines[2].replace('x_dim: ', '')
  y_mask_dim = lines[3].replace('y_mask_dim: ', '')
  train_test_id = lines[4].replace('train_test_id: ', '')

  best_epoch = None
  best_val_loss = 100
  for line in lines[5:]:
    if 'Epoch' in line:
      curr_epoch = int(line.split()[1].split('/')[0])
    if 'valid Loss' in line:
      val_loss = float(line.split()[-1])
      if val_loss < best_val_loss:
        best_epoch = curr_epoch
        best_val_loss = val_loss

  return best_epoch


model_dir = '/ahg/regevdata/projects/CRISPR-libraries/prj/lib-modeling/be_predict_bystander/params/'

models_design = pd.read_csv('models.csv', index_col = 0)

for idx, row in models_design.iterrows():
  editor_typ = 'AtoG' if 'ABE' in row['Internal base editor'] else 'CtoT'

  if 'ABE' in row['Internal base editor']:
    parent_dir = '/ahg/regevdata/projects/CRISPR-libraries/prj/lib-modeling/portable_modeling/out/c_ag_res_abe/'
  else:
    parent_dir = '/ahg/regevdata/projects/CRISPR-libraries/prj/lib-modeling/portable_modeling/out/c_ag_res/'

  model_nm = f"{row['Celltype']}_12kChar_{row['Internal base editor']}_{row['Model name']}"

  # Get log fn
  log_fn = parent_dir + '_log_' + model_nm + '.out'
  command = f'cp {log_fn} {model_dir}'
  subprocess.check_call(command, shell = True)

  best_epoch = get_epoch_from_log_fn(log_fn)

  # Get pytorch params
  param_fn = parent_dir + f"model_{model_nm}/model_epoch_{best_epoch}_statedict.pt"
  command = f'cp {param_fn} {model_dir}'
  subprocess.check_call(command, shell = True)

  print(model_nm)