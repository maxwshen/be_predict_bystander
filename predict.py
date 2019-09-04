# Model

from __future__ import absolute_import, division
from __future__ import print_function
import sys, string, pickle, subprocess, os, datetime, gzip, time
from collections import defaultdict, OrderedDict
import glob
import numpy as np, pandas as pd
import torch

#

nts = list('ACGT')
nt_to_idx = {nt: nts.index(nt) for nt in nts}
device = torch.device('cpu')
models_design = pd.read_csv('models.csv', index_col = 0)
model_dir = os.path.dirname(os.path.realpath(__file__)) + '/params/'
editor_profile_nt_cols = set()
core_substrate_nt = ''
model = None
tt_id = -1
init_flag = False
model_script = None

model_settings = {
  'celltype': None,
  'base_editor': None,
  '__base_editor_type': None,
  '__model_nm': None,
  '__param_epoch': None,
  '__combinatorial_central_pos': '6',
  # = 4 req. 0.1 s. 5 takes 0.4 seconds, 6 = 1.6 seconds, etc
  '__combinatorial_nt_limit': '5',   
  '__combinatorial_radii': '10',
  '__combinatorial_binary_start': '1',
  '__combinatorial_binary_end': '12',
}

model_nm_mapper = {}
for idx, row in models_design.iterrows():
  inp_set = (row['Public base editor'], row['Celltype'])
  model_nm = f"{row['Celltype']}_12kChar_{row['Internal base editor']}_{row['Model name']}"
  model_nm_mapper[inp_set] = model_nm


'''
  Usage: 
    import predict as bystander_model
    bystander_model.init_model(base_editor = '', celltype = '')

    pred_df = bystander_model.predict(seq)

  Supports base editors and cell-types described in models.csv.
  
'''

####################################################################
# Private
####################################################################

##
# Form query dataframe
##
def __form_single_muts_df(nt_cols, col_to_nt):
  '''
    Enumerate all single mutations at all editable positions
    Considers all mutations: N->N.
  '''
  dd = defaultdict(list)
  for col in nt_cols:
    ref_nt = col_to_nt[col]
    possible_muts = [nt for nt in nts if nt != ref_nt]
    other_cols = [c for c in nt_cols if c != col]
    for mut in possible_muts:
      dd[col].append(mut)

      for other_col in other_cols:
        dd[other_col].append(col_to_nt[other_col])
  return pd.DataFrame(dd)


def __form_combinatorial_core_df(nt_cols, col_to_nt):
  '''
    Enumerate combinations of edits at core editable positions.
    Considers all mutations: N->N.

    num_nt_limit: num. unique bases to form combinations of. 
      4^4 = 256
    dist_limit: position radii around central_pos to consider as core 
  '''
  # Find core to enumerate combinatorially
  # num_nt_limit = 5
  # dist_limit = 8
  # central_pos = 6
  num_nt_limit = int(model_settings['__combinatorial_nt_limit'])
  dist_limit = int(model_settings['__combinatorial_radii'])
  central_pos = int(model_settings['__combinatorial_central_pos'])

  def recurse_muts(num):
    # Consider all mutations
    if num == 1:
      return [['A'], ['C'], ['G'], ['T']]
    else:
      lists = recurse_muts(num - 1)
      new_lists = []
      for l in lists:
        for mut_nt in list('ACGT'):
          new_lists.append(l + [mut_nt])
      return new_lists

  dists = {}
  for col in nt_cols:
    if col[0] == core_substrate_nt:
      dist = abs(int(col[1:]) - central_pos)
      dists[col] = dist
  core_cols = sorted(dists, key = dists.get)[:num_nt_limit]
  core_cols = [ck for ck in core_cols if dists[ck] <= dist_limit]

  if len(core_cols) == 0:
    return pd.DataFrame()

  core_muts = recurse_muts(len(core_cols))
  core_muts = np.array(core_muts).T
  n = len(core_muts[0])
  dd = dict()
  for idx, col in enumerate(core_cols):
    dd[col] = list(core_muts[idx])

  for col in nt_cols:
    if col in core_cols:
      continue
    ref_nt = col_to_nt[col]
    dd[col] = [ref_nt] * n

  return pd.DataFrame(dd)


def __form_binary_combinatorial_core_df(nt_cols, col_to_nt):
  '''
    Enumerate combinations of edits at core editable positions.
    Considers only primary mutation: C->T. (or A->G)

    num_nt_limit: num. unique bases to form combinations of. 
      4^4 = 256
    dist_limit: position radii around central_pos to consider as core 
  '''
  start_pos = int(model_settings['__combinatorial_binary_start'])
  end_pos = int(model_settings['__combinatorial_binary_end'])
  allowed_pos = list(range(start_pos, end_pos + 1))

  if core_substrate_nt == 'C':
    edited_nt = 'T'
  elif core_substrate_nt == 'A':
    edited_nt = 'G'

  def recurse_muts(num):
    # Consider all mutations
    if num == 1:
      return [[core_substrate_nt], [edited_nt]]
    else:
      lists = recurse_muts(num - 1)
      new_lists = []
      for l in lists:
        for mut_nt in [core_substrate_nt, edited_nt]:
          new_lists.append(l + [mut_nt])
      return new_lists

  core_cols = []
  for col in nt_cols:
    if col[0] == core_substrate_nt:
      pos = int(col[1:])
      if pos in allowed_pos:
        core_cols.append(col)

  if len(core_cols) == 0:
    return pd.DataFrame()

  core_muts = recurse_muts(len(core_cols))
  core_muts = np.array(core_muts).T
  n = len(core_muts[0])
  dd = dict()
  for idx, col in enumerate(core_cols):
    dd[col] = list(core_muts[idx])

  for col in nt_cols:
    if col in core_cols:
      continue
    ref_nt = col_to_nt[col]
    dd[col] = [ref_nt] * n

  return pd.DataFrame(dd)


def __get_nt_cols(seq):
  nt_cols = []
  for idx in range(len(seq)):
    pos = idx - 19
    ref_nt = seq[idx]
    nt_col = f'{ref_nt}{pos}'
    if nt_col in editor_profile_nt_cols:
      nt_cols.append(nt_col)
  return nt_cols


def __seq_to_query_df(seq):
  '''
    No wild-type row since it's added during featurization.
  '''
  nt_cols = __get_nt_cols(seq)
  col_to_nt = {col: col[0] for col in nt_cols}

  single_mut_df = __form_single_muts_df(nt_cols, col_to_nt)
  combin_mut_df = __form_combinatorial_core_df(nt_cols, col_to_nt)
  combin_binary_mut_df = __form_binary_combinatorial_core_df(nt_cols, col_to_nt)

  query_df = single_mut_df.append(combin_mut_df, ignore_index = True, sort = False)
  query_df = query_df.append(combin_binary_mut_df, ignore_index = True, sort = False)
  query_df = query_df.drop_duplicates()

  # Filter wild-type
  query_df = query_df[query_df.apply(lambda row: sum([bool(col[0] == row[col]) for col in query_df.columns]) != len(query_df.columns), axis = 'columns')]

  return query_df


## 
# Init
##
def __init_editor_profile_nt_cols():
  '''
    Set global setting for current editor 
  '''
  global editor_profile_nt_cols
  editor_type = model_settings['__base_editor_type']

  editor_profile_df = pd.read_csv('editor_profiles.csv', index_col = 0)
  row = editor_profile_df.loc[editor_type]
  muts = editor_profile_df.columns

  for mut in muts:
    # Skip empty table entries
    if pd.isna(row[mut]):
      continue
    pos_range_str = row[mut]
    # Parse 'C to G'
    [ref_nt, obs_nt] = list(mut.replace(' to ', ''))
    # Parse '(-9, 20)'
    [pos_start, pos_end] = pos_range_str.replace('(', '').replace(')', '').split(', ')
    pos_start = int(pos_start)
    pos_end = int(pos_end)

    for pos in range(pos_start, pos_end + 1):
      nt_col = f'{ref_nt}{pos}'
      editor_profile_nt_cols.add(nt_col)

  return


def __init_editor_type():
  global core_substrate_nt
  global editor_type
  editor_nm = model_settings['base_editor']
  editor_type = models_design[models_design['Public base editor'] == editor_nm]['Base editor type'].iloc[0]
  core_substrate_nt = 'A' if editor_type == 'ABE' else 'C'
  model_settings['__base_editor_type'] = editor_type
  return


def __load_model_hyperparameters():
  log_fn = model_dir + '_log_' + model_settings['__model_nm'] + '.out'
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

  return model_hyperparameters, int(x_dim), int(y_mask_dim), int(train_test_id), best_epoch

####################################################################
# Public 
####################################################################

def predict(seq):
  assert len(seq) == 50, f'Error: Sequence provided is {len(seq)}, must be 50 (positions -19 to 30 w.r.t. gRNA (positions 1-20)'
  assert init_flag, f'Call .init_model() first.'
  seq = seq.upper()

  ## Call model
  query_df = __seq_to_query_df(seq)
  pred_df = query_df

  dataset = model_script.BaseEditing_Dataset(
    x = [seq], 
    y = [query_df], 
    nms = [0], 
    training = False
  )
  sample = dataset[0]

  with torch.no_grad():
    pred_log_probs = model(
      sample['x'],
      sample['y_mask'],
      sample['target'],
      sample['editable_index_info']
    )
  pred_probs = np.exp(pred_log_probs)
  pred_df['Predicted frequency'] = pred_probs
  pred_df = pred_df.sort_values(by = 'Predicted frequency', ascending = False)
  pred_df = pred_df.reset_index(drop = True)


  ## Get stats
  stats = {
    'Total predicted probability': sum(pred_df['Predicted frequency']),
    '50-nt target sequence': seq,
    'Assumed protospacer sequence': seq[20:40],
    'Celltype': model_settings['celltype'],
    'Base editor': model_settings['base_editor'],
  }

  return pred_df, stats


def init_model(base_editor = '', celltype = ''):
  # Check
  ok_editors = set(models_design['Public base editor'])
  assert base_editor in ok_editors, f'Bad base editor name\nAvailable options: {ok_editors}'
  ok_celltypes = set(models_design["Celltype"])
  assert celltype in ok_celltypes, f'Bad celltype\nAvailable options: {ok_celltypes}'

  # Update global settings
  spec = {
    'base_editor': base_editor,
    'celltype': celltype,
  }
  global model_settings
  for key in spec:
    if spec[key] != '':
      model_settings[key] = spec[key]

  # Init global parameters
  __init_editor_type()
  __init_editor_profile_nt_cols()

  model_settings['__model_nm'] = model_nm_mapper[(base_editor, celltype)]

  global model_script
  if model_settings['__base_editor_type'] == 'CBE':
    import model_CBE as model_script
  else:
    import model_ABE as model_script

  # Load model
  package = __load_model_hyperparameters()
  model_hyperparameters, x_dim, y_mask_dim, train_test_id, best_epoch = package
  model_settings['__param_epoch'] = best_epoch
  model_script.parse_custom_hyperparams(model_hyperparameters)

  global tt_id
  tt_id = train_test_id

  global model
  model = model_script.DeepAutoregressiveModel(x_dim, y_mask_dim)
  model.load_state_dict(torch.load(
    model_dir + f"model_{model_settings['__model_nm']}_epoch_{model_settings['__param_epoch']}_statedict.pt"
  ))
  model.eval()

  print(f'Model successfully initialized. Settings:')
  public_settings = [key for key in model_settings if key[:2] != '__']
  for key in public_settings:
    print(f'\t{key}: {model_settings[key]}')

  global init_flag
  init_flag = True
  return

