# Model

from __future__ import absolute_import, division
from __future__ import print_function
import sys, string, pickle, subprocess, os, datetime, gzip, time
from collections import defaultdict, OrderedDict

import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.utils
import torch.nn as nn

import glob

import numpy as np, pandas as pd
np.random.seed(seed = 0)
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

nts = list('ACGT')
nt_to_idx = {nt: nts.index(nt) for nt in nts}

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

random_seed = 0
torch.manual_seed(random_seed)

hyperparameters = {
  # featurization params
  'context_feature': True,
  'fullcontext_feature': False,
  'position_feature': True,
  'context_radii': 7,

  # architecture
  # 'encoder_hidden_sizes': [64, 64],
  'encoder_hidden_sizes': [16, 16],
  'decoder_hidden_sizes': [64, 64, 64, 64],
  'dropout_p': 0.05,

  # learning params
  'learning_rate': 2e-4,
  'exponential_lr_decay': 1,
  'plateau_patience': 5,
  'plateau_threshold': 1e-3,
  'plateau_factor': 0.5,

  # 'batch_size': 1,
  'batch_size': 5,
  'num_epochs': 300,
}

fold_nm = ''

'''
  TODO: Implement ABE vs. CBE behavior, this script is for CBE originally
'''

##
# Support
##
def parse_custom_hyperparams(custom_hyperparams):
  # Defaults
  global hyperparameters

  if custom_hyperparams == '':
    return

  # Parse hyperparams
  for term in custom_hyperparams.split('+'):
    [kw, args] = term.split(':')
    if kw in ['encoder_hidden_sizes', 'decoder_hidden_sizes']:
      # Expect comma-separated ints
      parse = lambda arg: [int(s) for s in arg.split(',')]
    if kw in ['context_feature', 'fullcontext_feature', 'position_feature']:
      # Expect 1 or 0
      parse = lambda arg: bool(int(arg))
    if kw in ['context_radii']:
      parse = lambda arg: int(arg)
    if kw in ['learning_rate', 'plateau_patience', 'plateau_threshold', 'dropout_p']:
      parse = lambda arg: float(arg)
    if kw in hyperparameters:
      hyperparameters[kw] = parse(args)
  return


##
# Model
##
class DeepAutoregressiveModel(nn.Module):
  def __init__(self, x_dim, y_mask_dim):
    super().__init__()
    self.encoder_Fs = self.init_encoder(x_dim)

    enc_last_layer_size = hyperparameters['encoder_hidden_sizes'][-1]
    self.decoder_Fs = self.init_decoder(enc_last_layer_size + y_mask_dim)

    self.unedited_bias = torch.nn.Parameter(
      torch.nn.init.xavier_uniform_(torch.randn(2, 30))
    )

  def unedited_biaser(self, input, editable_index_info):
    '''
      input.shape: (
        (n.uniq.e + 1, n.edit.b, 1, 4)
      )
      output.shape: same

      Need: 
        1. index of editable base -> position.
        2. index of editable base -> ref nt.
      Use this to extract position-wise bias to add
      to input.
    '''
    params = self.unedited_bias

    def form_single(pos, ref_nt):
      ref_nt_indexer = {'A': 0, 'C': 1}
      ref_nt_index = ref_nt_indexer[ref_nt]
      pos_idx = pos + 9
      single_vec = torch.zeros(1, 4)
      single_vec[0][nts.index(ref_nt)] = params[ref_nt_index][pos_idx]
      # shape: (1, 4)
      return single_vec

    bias_tensor = [
      form_single(
        editable_index_info['pos'][idx],
        editable_index_info['ref_nt'][idx],
      ) for idx in editable_index_info['pos']
    ]
    # shape: (n.edit.b, 1, 4)
    bias_tensor = torch.stack(bias_tensor)

    # shape: (n.uniq.e + 1, n.edit.b, 1, 4)
    bias_tensor = bias_tensor.expand(
      input.shape[0],
      bias_tensor.shape[0],
      bias_tensor.shape[1],
      bias_tensor.shape[2],
    )
    bias_tensor = bias_tensor.to(device)
    return input + bias_tensor
  
  def init_encoder(self, input_size):
    layer_sizes = [input_size] + hyperparameters['encoder_hidden_sizes']
    self.num_encoder_layers = len(layer_sizes) - 1
    modules = []
    for layer_idx, (i, o) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
      fs = {
        'linear': nn.Linear(i, o),
        'norm': nn.LayerNorm(o),
        'activation': nn.ReLU(),
        'dropout': nn.Dropout(p = hyperparameters['dropout_p'])
      }
      for nm in fs:
        module = fs[nm]
        name = f'{nm}_{layer_idx}'
        modules.append([name, module])
    return torch.nn.ModuleDict(modules)

  def init_decoder(self, input_size):
    layer_sizes = [input_size] + hyperparameters['decoder_hidden_sizes'] + [4]
    self.num_decoder_layers = len(layer_sizes) - 1
    modules = []
    for layer_idx, (i, o) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
      fs = {
        'linear': nn.Linear(i, o),
        'norm': nn.LayerNorm(o),
        'activation': nn.ReLU(),
        'dropout': nn.Dropout(p = hyperparameters['dropout_p'])
      }
      for nm in fs:
        module = fs[nm]
        name = f'{nm}_{layer_idx}'
        modules.append([name, module])
    return torch.nn.ModuleDict(modules)

  def get_function_info(self, f_nm):
    w = f_nm.split('_')
    f_info = {
      'type': w[0],
      'input_dim': int(w[1]),
      'output_dim': int(w[2]),
      'layer_num': int(w[3]),
    }
    return f_info

  def decoder(self, input):
    # Residual block
    output = input
    layer_nums = list(range(self.num_decoder_layers))
    last_layer = layer_nums[-1]
    for layer_idx in layer_nums:
      linear_f = self.decoder_Fs[f'linear_{layer_idx}']
      norm_f = self.decoder_Fs[f'norm_{layer_idx}']
      act_f = self.decoder_Fs[f'activation_{layer_idx}']
      drop_f = self.decoder_Fs[f'dropout_{layer_idx}']

      identity = output
      output = linear_f(output)
      if layer_idx != last_layer:
        output = norm_f(output)
        output = act_f(output)
        # Consider another linear_f
        if output.shape == identity.shape:
          output += identity
        output = drop_f(output)

    return output

  def encoder(self, input):
    # Residual block
    output = input
    layer_nums = list(range(self.num_encoder_layers))
    last_layer = layer_nums[-1]
    for layer_idx in layer_nums:
      linear_f = self.encoder_Fs[f'linear_{layer_idx}']
      norm_f = self.encoder_Fs[f'norm_{layer_idx}']
      act_f = self.encoder_Fs[f'activation_{layer_idx}']
      drop_f = self.encoder_Fs[f'dropout_{layer_idx}']

      identity = output
      output = linear_f(output)
      if layer_idx != last_layer:
        output = norm_f(output)
        output = act_f(output)
        # Consider another linear_f
        if output.shape == identity.shape:
          output += identity
        output = drop_f(output)

    return output


  def forward(self, x, y_mask, target, editable_index_info):
    '''
      Forward pass for a single target site.

      Shapes;
        n.uniq.e = num. unique obs. edits, depends on seq.
        n.edit.b = num. editable bases, depends on seq.

      x.shape = (n.edit.b, x_dim)
      y_mask.shape = (n.uniq.e + 1, n.edit.b, y_mask_dim)
      target.shape = (n.uniq.e + 1, n.edit.b, 4, 1)
      obs_freq.shape = (n.uniq.e)

      n.uniq.e + 1 is due to including a row for wild-type, which is used to adjust all predicted probabilities by (1 - wild-type) denominator.
    '''

    # 1. Model encodes x -> (n.edit.b, x_enc_dim)
    enc_x = self.encoder(x)
    x_shape = enc_x.shape

    # 2. Expanding and catting with y_mask ->
    #   (n.uniq.e + 1, n.edit.b, x_enc_dim + y_mask_dim)
    expand_x = enc_x.expand(
      y_mask.shape[0],
      x_shape[0],
      x_shape[1],
    )
    y_inp = torch.cat((expand_x, y_mask), dim = -1)

    # 3. Decode -> (n.uniq.e + 1, n.edit.b, 1, 4)
    y_out = self.decoder(y_inp)
    y_out = y_out.reshape(
      y_out.shape[0],
      y_out.shape[1],
      1,
      y_out.shape[2],
    )

    # 4. Add unedited bias, then softmax -> (n.uniq.e + 1, n.edit.b, 1, 4)
    y_out = self.unedited_biaser(y_out, editable_index_info)
    y_out = F.log_softmax(y_out, dim = -1)

    # 5. Matmul with target one-hot-encoding ->
    #   (n.uniq.e + 1, n.edit.b, 1, 1), reshape ->
    #   (n.uniq.e + 1, n.edit.b)
    lls = torch.matmul(y_out, target)
    lls = lls.reshape(lls.shape[:2])

    # 6. Sum log likelihoods -> (n.uniq.e + 1)
    lls = torch.sum(lls, dim = -1)

    # 7. Adjust all likelihoods by (1 - wild-type) denominator -> (n.uniq.e). Wild-type encoded at last position.
    one_minus_wildtype_log_prob = torch.log(1 - torch.exp(lls[-1]))
    lls = lls[:-1] - one_minus_wildtype_log_prob

    return lls


##
# Data class featurization
##
class BaseEditing_Dataset(Dataset):
  '''
    X: list of 50-nt sequences
    Y: list of dataframes
      Columns: Editable nucleotides and positions, frequency
      Rows: Unique editing outcomes at the target site
    
    Transform X and Y into lists of tensors.

    At prediction-time, generate Y using heuristic from X only (no need for obs_freq).

    Shapes;
      N = number of target sites
      n.uniq.e = num. unique obs. edits, depends on seq.
      n.edit.b = num. editable bases, depends on seq.

    x.shape = (N, n.edit.b, x_dim)
    y_mask.shape = (N, n.uniq.e + 1, n.edit.b, y_mask_dim)
    target.shape = (N, n.uniq.e + 1, n.edit.b, 4, 1)
    obs_freq.shape = (N, n.uniq.e)

    n.uniq.e + 1 is due to including a row for wild-type, which is used to adjust all predicted probabilities by (1 - wild-type) denominator.
  '''
  def __init__(self, x, y, nms, training = True):
    self.init_edit_encodings()
    self.init_nt_cols(y)

    self.nms = nms

    x, x_dim = self.featurize(x)
    self.x = x
    self.x_dim = x_dim

    package = self.to_autoregress_masked_tensors(y)
    y_mask, y_mask_dim, target, editable_index_info = package
    self.y_mask = y_mask
    self.y_mask_dim = y_mask_dim
    self.target = target
    self.editable_index_info = editable_index_info

    if training:
      self.obs_freq = self.get_obs_freqs(y)
    else:
      self.obs_freq = defaultdict(lambda: -1)


  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return {
      'x' : self.x[idx],
      'y_mask' : self.y_mask[idx],
      'target': self.target[idx],
      'obs_freq': self.obs_freq[idx],
      'editable_index_info': self.editable_index_info[idx],
      'nm': self.nms[idx],
    }

  ##
  # Featurize X
  ##
  def featurize(self, X):
    # x provided is 50-nt, but we care only about center 30-nt.
    # 30-nt ranges from positions -9 to 20 relative to gRNA.
    ftx = []
    offset = 10
    print('Featurizing X')
    # timer = _util.Timer(total = len(X))
    for _, seq in enumerate(X):

      nt_cols = self.all_nt_cols[_]
      editable_pos = sorted([int(col[1:]) for col in nt_cols])
      seq_30nt = seq[10:-10]

      # TO DO: subset to only editable positions
      single_target_ftx = []
      for pos in editable_pos:
        idx = pos + 9
        # use offset_idx to query seq at current pos
        offset_idx = offset + idx

        assert len(seq) == 30 + offset * 2, 'Bad offset'

        X_singlepos = []
        if hyperparameters['context_feature'] == True:
          radii = hyperparameters['context_radii']
          X_singlepos += self.ohe_seq(seq[offset_idx - radii : offset_idx + radii + 1])
        
        if hyperparameters['fullcontext_feature'] == True:
          X_singlepos += self.ohe_seq(seq_30nt)

        if hyperparameters['position_feature'] == True:
          X_singlepos += self.ohe_position(pos, -9, 20)

        single_target_ftx.append(X_singlepos)

      # single_target_ftx.shape = (num. editable bases, x_dim)
      ftx.append(torch.Tensor(single_target_ftx))
      # timer.update()

    x_dim = ftx[0].shape[-1]
    return ftx, x_dim

  def ohe_position(self, pos, min_pos, max_pos):
    encode = [0] * (max_pos - min_pos + 1)
    encode[pos - min_pos] = 1
    return encode

  def ohe_seq(self, seq):
    encoder = {
      'A': [1, 0, 0, 0],
      'C': [0, 1, 0, 0],
      'G': [0, 0, 1, 0],
      'T': [0, 0, 0, 1],
    }
    encode = []
    for nt in seq:
      encode += encoder[nt]
    return encode

  ##
  # obs freqs
  ##
  def get_obs_freqs(self, Y):
    '''
      List of tensors with shape: (
        num. unique obs. edits,
      )
    '''
    obs_freqs = []
    print('Getting obs freqs...')
    # timer = _util.Timer(total = len(Y))
    for _, y in enumerate(Y):
      freqs = torch.Tensor(list(y['Y']))
      obs_freqs.append(freqs)
      # timer.update()
    return obs_freqs

  ##
  # Y
  ##
  def to_autoregress_masked_tensors(self, Y):
    tensors_Y = []
    tensors_target = []
    editable_index_info = []
    print('Transforming Y into tensors...')
    # timer = _util.Timer(total = len(Y))
    for idx, y in enumerate(Y):
      single_target_y = []
      single_target_targets = []

      nt_cols = self.all_nt_cols[idx]
      editable_pos_to_nt = {int(col[1:]): col[0] for col in nt_cols}
      editable_pos = sorted(list(editable_pos_to_nt.keys()))
      pos_to_col = {int(col[1:]): col for col in nt_cols}
      ref_nts = [nt_col[0] for nt_col in nt_cols]

      single_target_editable_info = {
        'pos': {idx: editable_pos[idx] for idx in range(len(editable_pos))},
        'ref_nt': {idx: pos_to_col[editable_pos[idx]][0] for idx in range(len(editable_pos))},
      }
      editable_index_info.append(single_target_editable_info)  

      # Append wild-type row
      wt_row = pd.DataFrame({col: col[0] for col in nt_cols}, index = [0])
      y = y.append(wt_row, ignore_index = True, sort = False)

      for jdx, row in y.iterrows():
        col_to_obs_edit = {col: row[col] for col in nt_cols}
        single_row_y = self.form_masked_edit_vectors(
          editable_pos,
          pos_to_col,
          col_to_obs_edit,
        )
        single_target_y.append(single_row_y)

        single_row_target = self.form_target_vectors(
          editable_pos,
          pos_to_col,
          col_to_obs_edit,
        )
        single_target_targets.append(single_row_target)

      '''
        single_target_y.shape = (
          num. unique edits + 1, 
          num. editable bases, 
          y_mask_dim
        )
      '''
      tensors_Y.append(torch.Tensor(single_target_y))
      tensors_target.append(torch.Tensor(single_target_targets))
      # timer.update()

    y_mask_dim = tensors_Y[0].shape[-1]
    return tensors_Y, y_mask_dim, tensors_target, editable_index_info

  def init_edit_encodings(self):
    '''
      Encoding decisions
      - Reverse complement G->A should be the same as C->T
      - No edit is all 0s
    '''
    self.edit_mapper = dict()
    self.edit_mapper['A'] = {
      'A': [0, 0, 0],
      'C': [1, 0, 0],
      'G': [0, 1, 0],
      'T': [0, 0, 1],
    }
    self.edit_mapper['C'] = {
      'A': [1, 0, 0],
      'C': [0, 0, 0],
      'G': [0, 1, 0],
      'T': [0, 0, 1],
    }
    self.uneditable_vec = [0.33, 0.33, 0.33]
    self.future_mask_vec = [-1, -1, -1]
    self.ohe_len = 3
    return

  def form_masked_edit_vectors(self, editable_pos, pos_to_col, col_to_obs_edit):
    # Form least masked edit vector
    least_masked_pos = max(editable_pos)
    least_masked_edit_vector_a = []
    least_masked_edit_vector_c = []
    for p in range(-9, 20 + 1):
      if p < least_masked_pos:
        if p in editable_pos:
          # Editable
          col = pos_to_col[p]
          ref_nt = col[0]
          obs_nt = col_to_obs_edit[col]
          if ref_nt == 'A':
            least_masked_edit_vector_a += self.edit_mapper[ref_nt][obs_nt]
            least_masked_edit_vector_c += self.uneditable_vec
          elif ref_nt == 'C':
            least_masked_edit_vector_c += self.edit_mapper[ref_nt][obs_nt]
            least_masked_edit_vector_a += self.uneditable_vec
        else:
          # Uneditable
          least_masked_edit_vector_a += self.uneditable_vec
          least_masked_edit_vector_c += self.uneditable_vec
      elif p >= least_masked_pos:
        # Masked current and future
        least_masked_edit_vector_a += self.future_mask_vec
        least_masked_edit_vector_c += self.future_mask_vec

    # Produce all edit vecs by adding masking to least masked vec
    single_row_y = []
    for pos in editable_pos:
      idx = pos + 9   # hide current pos
      mask_len = 30 - idx
      mask = self.future_mask_vec * mask_len
      mask_idx = idx * self.ohe_len
      # print(pos, len(least_masked_edit_vector_a[:mask_idx]), len(mask))
      edit_vector_c = least_masked_edit_vector_a[:mask_idx] + mask
      edit_vector_g = least_masked_edit_vector_c[:mask_idx] + mask
      edit_vec = edit_vector_c + edit_vector_g 
      single_row_y.append(edit_vec)

    return single_row_y

  def form_target_vectors(self, editable_pos, pos_to_col, col_to_obs_edit):
    target_vec = []
    for pos in editable_pos:
      obs_nt = col_to_obs_edit[pos_to_col[pos]]
      # shape: (4, 1)
      # ohe = torch.Tensor([self.ohe_seq(obs_nt)]).transpose(0, 1)
      ohe = [[s] for s in self.ohe_seq(obs_nt)]
      target_vec.append(ohe)
    return target_vec

  ##
  # Helper
  ##
  def get_nt_cols(self, df):
    nt_cols = []
    for col in df.columns:
      if 'Count' in col or 'Frequency' in col or col == 'Y':
        continue
      nt_cols.append(col)
    return nt_cols

  def init_nt_cols(self, Y):
    self.all_nt_cols = dict()
    for idx, y in enumerate(Y):
      self.all_nt_cols[idx] = self.get_nt_cols(y)
    return
