# Model

from __future__ import absolute_import, division
from __future__ import print_function
import sys, string, pickle, subprocess, os, datetime, gzip, time
from collections import defaultdict, OrderedDict
# import _config, _data, _util

import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
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

def copy_model_script():
  from shutil import copyfile
  copyfile(__file__, model_dir + __file__)

def check_num_models():
  import glob
  dirs = glob.glob(out_dir + 'model*')
  return len(dirs)

def print_and_log(text):
  with open(log_fn, 'a') as f:
    f.write(text + '\n')
  print(text)
  return

def create_model_dir():
  num_existing = check_num_models()
  global model_dir
  if fold_nm == '':
    run_id = str(num_existing + 1)
  else:
    run_id = fold_nm

  model_dir = out_dir + 'model_' + run_id + '/'
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  print('Saving model in ' + model_dir)

  global log_fn
  log_fn = out_dir + '_log_%s.out' % (run_id)
  with open(log_fn, 'w') as f:
    pass
  print_and_log('model dir: ' + model_dir)
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
      ref_nt_indexer = {'C': 0, 'G': 1}
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
# Training
##
def train_model(model, optimizer, schedulers, datasets, dataset_sizes):
  writer = SummaryWriter(model_dir)
  since = time.time()

  loss_func = nn.KLDivLoss(reduction = 'batchmean')

  bs = hyperparameters['batch_size']
  num_epochs = hyperparameters['num_epochs']
  epoch_loss = dict()
  for epoch in range(num_epochs):
    print_and_log('-' * 10)
    print_and_log('Epoch %s/%s at %s' % (epoch, num_epochs - 1, datetime.datetime.now()))

    # Each epoch: training phase, validation phase
    for phase in ['train', 'valid']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      dataset = datasets[phase]
      N = dataset_sizes[phase]
      random_idxs = np.arange(N)
      np.random.shuffle(random_idxs)

      running_loss = 0.0
      with torch.set_grad_enabled(phase == 'train'):
        for idx in range((N - 1) // bs + 1):
          batch_start_idx = idx * bs
          batch_end_idx = min(idx * bs + bs, N - 1)

          if bs > 1:
            batch_loss = torch.autograd.Variable(
              torch.zeros(1).to(device), 
              requires_grad = True
            )
          batch_idxs = random_idxs[batch_start_idx : batch_end_idx]
          if len(batch_idxs) == 0:
            continue
          for sample_idx in batch_idxs:
            # Single target site with multiple unique editing outcomes (serves as batch for GPU).
            sample = dataset[sample_idx]

            x = sample['x'].to(device)
            y_mask = sample['y_mask'].to(device)
            target = sample['target'].to(device)
            obs_freq = sample['obs_freq'].to(device)
            editable_index_info = sample['editable_index_info']

            pred_log_probs = model(x, y_mask, target, editable_index_info)

            '''
              bs = "batch size" = 1
              pred_log_probs.shape = (bs, n.uniq.e)
              obs_freq.shape = (bs, n.uniq.e)
            '''
            pred_log_probs = pred_log_probs.reshape(1, pred_log_probs.shape[0])
            obs_freq = obs_freq.reshape(1, obs_freq.shape[0])

            loss = loss_func(pred_log_probs, obs_freq)
            running_loss += loss.item()

            if bs > 1:
              batch_loss = batch_loss + loss
            else:
              batch_loss = loss

          # Step once per batch
          if phase == 'train':
            batch_loss.backward()
            optimizer.step()
            schedulers['exponential'].step()
            optimizer.zero_grad()
            del batch_loss

      epoch_loss[phase] = running_loss / dataset_sizes[phase]

      print_and_log('{} Loss: {:.4f}'.format(phase, epoch_loss[phase]))
      writer.add_scalar(phase + '_loss', epoch_loss[phase], epoch)

      if phase == 'valid':
        schedulers['cyclic'].step(epoch_loss[phase])
        # schedulers['plateau'].step(epoch_loss[phase])

    # Each epoch
    torch.save(model, model_dir + 'model_epoch_%s_entiremodel.pt' % (epoch))
    torch.save(model.state_dict(), model_dir + 'model_epoch_%s_statedict.pt' % (epoch))

  time_elapsed = time.time() - since
  print_and_log('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

  torch.save(model.state_dict(), model_dir + 'model_final.pt')
  writer.close()
  return


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
    self.edit_mapper['C'] = {
      'A': [1, 0, 0],
      'C': [0, 0, 0],
      'G': [0, 1, 0],
      'T': [0, 0, 1],
    }
    self.edit_mapper['G'] = {
      'A': [0, 0, 1],
      'C': [0, 1, 0],
      'G': [0, 0, 0],
      'T': [1, 0, 0],
    }
    self.uneditable_vec = [0.33, 0.33, 0.33]
    self.future_mask_vec = [-1, -1, -1]
    self.ohe_len = 3
    return

  def form_masked_edit_vectors(self, editable_pos, pos_to_col, col_to_obs_edit):
    # Form least masked edit vector
    least_masked_pos = max(editable_pos)
    least_masked_edit_vector_c = []
    least_masked_edit_vector_g = []
    for p in range(-9, 20 + 1):
      if p < least_masked_pos:
        if p in editable_pos:
          # Editable
          col = pos_to_col[p]
          ref_nt = col[0]
          obs_nt = col_to_obs_edit[col]
          if ref_nt == 'C':
            least_masked_edit_vector_c += self.edit_mapper[ref_nt][obs_nt]
            least_masked_edit_vector_g += self.uneditable_vec
          elif ref_nt == 'G':
            least_masked_edit_vector_g += self.edit_mapper[ref_nt][obs_nt]
            least_masked_edit_vector_c += self.uneditable_vec
        else:
          # Uneditable
          least_masked_edit_vector_c += self.uneditable_vec
          least_masked_edit_vector_g += self.uneditable_vec
      elif p >= least_masked_pos:
        # Masked current and future
        least_masked_edit_vector_c += self.future_mask_vec
        least_masked_edit_vector_g += self.future_mask_vec

    # Produce all edit vecs by adding masking to least masked vec
    single_row_y = []
    for pos in editable_pos:
      idx = pos + 9   # hide current pos
      mask_len = 30 - idx
      mask = self.future_mask_vec * mask_len
      mask_idx = idx * self.ohe_len
      # print(pos, len(least_masked_edit_vector_c[:mask_idx]), len(mask))
      edit_vector_c = least_masked_edit_vector_c[:mask_idx] + mask
      edit_vector_g = least_masked_edit_vector_g[:mask_idx] + mask
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

##
# Load data
##
def load_data(dataset_id, train_test_id):
  # Load data, check for preprocessed.
  # Dataset pickle ~ 1GB
  # preprocessed_data_fn = out_dir + 'preprocessed_dataset_%s.pkl' % (dataset_id)
  # if os.path.exists(preprocessed_data_fn):
  #   print('Loading data from pickle...')
  #   with open(preprocessed_data_fn, 'rb') as f:
  #     package = pickle.load(f)
  # else:
  #   X, Y, NAMES = load_human_data(dataset_id)
  #   package = generate_train_test(X, Y, NAMES, dataset_id)
  #   print('Saving preprocessed data to pickle...')
  #   with open(preprocessed_data_fn, 'wb') as f:
  #     pickle.dump(package, f)

  X, Y, NAMES = load_human_data(dataset_id)

  # print('Subsetting to 100 for testing..')
  # X = X[:100]
  # Y = Y[:100]
  # NAMES = NAMES[:100]

  package = generate_train_test(X, Y, NAMES, dataset_id, train_test_id)

  datasets, dataset_sizes, x_dim, y_mask_dim = package
  return datasets, dataset_sizes, x_dim, y_mask_dim

def load_human_data(dataset_id):
  import _config, _data, _util

  lib_nm = _data.get_lib_nm(dataset_id)
  lib_design, seq_col = _data.get_lib_design(dataset_id)
  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}
  nm_to_p0idx = {nm: p0idx for nm, p0idx in zip(nms, lib_design['Protospacer position zero index'])}

  Y_dir = _config.OUT_PLACE + 'combin_data_Y_imputewt/'
  with gzip.open(Y_dir + '%s.pkl.gz' % (dataset_id), 'rb') as f:
    Y = pickle.load(f)
  
  NAMES = list(Y.keys())
  Y = list(Y.values())

  # Load X
  X = []
  timer = _util.Timer(total = len(NAMES))
  for nm, y in zip(NAMES, Y):
    seq = nm_to_seq[nm]
    p0idx = nm_to_p0idx[nm]

    if p0idx - 9 - 10 < 0:
      # CtoT, AtoG, CtoGA libraries -- add fixed prefix
      prefix = 'GATGGGTGCGACGCGTCAT'
      p0idx = p0idx + len(prefix)
    else:
      prefix = ''
    seq = prefix + seq

    if p0idx + 20 + 10 + 1 > len(seq):
      seq = seq + 'AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT'

    # seq_30nt = seq[p0idx - 9 : p0idx + 20 + 1]
    seq_50nt = seq[p0idx - 9 - 10 : p0idx + 20 + 10 + 1]
    assert len(seq_50nt) == 50
    X.append(seq_50nt)

  return X, Y, NAMES

def generate_train_test(X, Y, NAMES, dataset_id, train_test_id, valid_frac = 0.10):
  lib_nm = _data.get_lib_nm(dataset_id)
  tt_df = pd.read_csv(_config.OUT_PLACE + 'gen_traintest_idxs/%s_%s.csv' % (lib_nm, train_test_id), index_col = 0)
  nms_train = set(tt_df[tt_df['Category'] == 'Train']['Name'])
  nms_test = set(tt_df[tt_df['Category'] == 'Test']['Name'])

  train_idxs = [NAMES.index(nm) for nm in nms_train if nm in NAMES]
  test_idxs = [NAMES.index(nm) for nm in nms_test if nm in NAMES]

  # Validation set is last % of training set
  num_valid = int(len(train_idxs) * valid_frac)
  valid_idxs = train_idxs[-num_valid:]
  train_idxs = train_idxs[:-num_valid]

  print(f'Training set size: {len(train_idxs)}')
  print(f'Validation set size: {len(valid_idxs)}')
  print(f'Test set size: {len(test_idxs)}')
  print(f'Total size: {len(train_idxs) + len(valid_idxs) + len(test_idxs)}')

  X_train = [X[idx] for idx in train_idxs]
  X_valid = [X[idx] for idx in valid_idxs]
  X_test = [X[idx] for idx in test_idxs]

  Y_train = [Y[idx] for idx in train_idxs]
  Y_valid = [Y[idx] for idx in valid_idxs]
  Y_test = [Y[idx] for idx in test_idxs]

  NAMES_train = [NAMES[idx] for idx in train_idxs]
  NAMES_valid = [NAMES[idx] for idx in valid_idxs]
  NAMES_test = [NAMES[idx] for idx in test_idxs]


  datasets = {
    'train': BaseEditing_Dataset(x = X_train, y = Y_train, nms = NAMES_train),
    'valid': BaseEditing_Dataset(x = X_valid, y = Y_valid, nms = NAMES_valid),
    'test': BaseEditing_Dataset(x = X_test, y = Y_test, nms = NAMES_test),
  }
  x_dim = datasets['train'].x_dim
  y_mask_dim = datasets['train'].y_mask_dim

  dataset_sizes = {
    'train': len(X_train),
    'valid': len(X_valid),
    'test': len(X_test),
  }

  return datasets, dataset_sizes, x_dim, y_mask_dim

##
# Main
##
def main(dataset_id = 'mES_12kChar_eA3A', custom_fold_nm = '', train_test_id = '0', custom_hyperparams = ''):
  global fold_nm
  if custom_fold_nm != '':
    fold_nm = f'{dataset_id}_{train_test_id}_{custom_fold_nm}'
  else:
    fold_nm = ''

  parse_custom_hyperparams(custom_hyperparams)

  datasets, dataset_sizes, x_dim, y_mask_dim = load_data(dataset_id, train_test_id)

  model = DeepAutoregressiveModel(x_dim, y_mask_dim).to(device)

  for param in model.parameters():
    print(type(param.data), param.shape)

  # optimizer = torch.optim.Adam(
  #   model.parameters(), 
  #   lr = hyperparameters['learning_rate'],
  # )
  optimizer = torch.optim.SGD(
    model.parameters(), 
    lr = hyperparameters['learning_rate'],
    momentum = 0.5,
  )

  schedulers = {
    'exponential': torch.optim.lr_scheduler.ExponentialLR(
      optimizer, 
      hyperparameters['exponential_lr_decay'],
    ),
    # 'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
    #   optimizer,
    #   patience = hyperparameters['plateau_patience'],
    #   threshold = hyperparameters['plateau_threshold'],
    #   factor = hyperparameters['plateau_factor'],
    #   verbose = True,
    #   threshold_mode = 'abs',
    # )
    'cyclic': torch.optim.lr_scheduler.CyclicLR(
      optimizer,
      base_lr = 1e-6,
      max_lr = 2e-4,
      mode = 'triangular2',
    )
  }

  create_model_dir()
  copy_model_script()
  print_and_log(f'Hyperparameters: {custom_hyperparams}')
  print_and_log(f'x_dim: {x_dim}')
  print_and_log(f'y_mask_dim: {y_mask_dim}')
  print_and_log(f'train_test_id: {train_test_id}')

  train_model(
    model, 
    optimizer, 
    schedulers,
    datasets, 
    dataset_sizes, 
  )

  return

def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  _util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  # exp_nms = [
  #   'mES_12kChar_eA3A',
  # ]

  exp_nms = [
    # 'HEK293T_12kChar_ABE',
    # 'HEK293T_12kChar_ABE-CP1040',
    'HEK293T_12kChar_AID',
    'HEK293T_12kChar_BE4',
    'HEK293T_12kChar_BE4-CP1028',
    'HEK293T_12kChar_CDA',
    'HEK293T_12kChar_eA3A',
    'HEK293T_12kChar_evoAPOBEC',
    # 'mES_12kChar_ABE',
    # 'mES_12kChar_ABE-CP1040',
    'mES_12kChar_AID',
    'mES_12kChar_BE4',
    'mES_12kChar_BE4-CP1028',
    'mES_12kChar_CDA',
    'mES_12kChar_eA3A',
    'mES_12kChar_evoAPOBEC',
  ]

  run_prefix = '190524'

  hyperparam_combinations = {
    'enc_64x2_dec_256x2_nc_r5_d0.05': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:256,256+context_feature:1+fullcontext_feature:0+position_feature:1+context_radii:5+dropout_p:0.05',
    'enc_64x2_dec_256x2_nc_r5_nopos_d0.05': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:256,256+context_feature:1+fullcontext_feature:0+position_feature:0+context_radii:5+dropout_p:0.05',


    # # baselines
    # 'og_vanilla': 'encoder_hidden_sizes:16,16+decoder_hidden_sizes:16,16+context_feature:1+fullcontext_feature:0+position_feature:1+context_radii:3+dropout_p:0.0',
    # 'og_64x2_fullcontext_r5': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:64,64+context_feature:1+fullcontext_feature:1+position_feature:1+context_radii:5+dropout_p:0.0',

    # # nc
    # '32x2_nc_r5_d0.05': 'encoder_hidden_sizes:32,32+decoder_hidden_sizes:32,32+context_feature:1+fullcontext_feature:0+position_feature:1+context_radii:5+dropout_p:0.05',
    # '64x2_nc_r5_d0.05': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:64,64+context_feature:1+fullcontext_feature:0+position_feature:1+context_radii:5+dropout_p:0.05',
    # '128x2_nc_r5_d0.05': 'encoder_hidden_sizes:128,128+decoder_hidden_sizes:128,128+context_feature:1+fullcontext_feature:0+position_feature:1+context_radii:5+dropout_p:0.05',
    # '256x2_nc_r5_d0.05': 'encoder_hidden_sizes:256,256+decoder_hidden_sizes:256,256+context_feature:1+fullcontext_feature:0+position_feature:1+context_radii:5+dropout_p:0.05',

    # # Extended nc
    # 'enc_64x2_dec_64x5_nc_r5_d0.05': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:64,64,64,64,64+context_feature:1+fullcontext_feature:0+position_feature:1+context_radii:5+dropout_p:0.05',
    # 'enc_64x2_dec_128x2_nc_r5_d0.05': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:128,128+context_feature:1+fullcontext_feature:0+position_feature:1+context_radii:5+dropout_p:0.05',
    # 'enc_64x2_dec_256x2_nc_r5_d0.05': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:256,256+context_feature:1+fullcontext_feature:0+position_feature:1+context_radii:5+dropout_p:0.05',

    # # nopos
    # 'og_vanilla_nopos': 'encoder_hidden_sizes:16,16+decoder_hidden_sizes:16,16+context_feature:1+fullcontext_feature:0+position_feature:0+context_radii:3+dropout_p:0.0',
    # '32x2_nc_r5_nopos_d0.05': 'encoder_hidden_sizes:32,32+decoder_hidden_sizes:32,32+context_feature:1+fullcontext_feature:0+position_feature:0+context_radii:5+dropout_p:0.05',
    # '64x2_nc_r5_nopos_d0.05': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:64,64+context_feature:1+fullcontext_feature:0+position_feature:0+context_radii:5+dropout_p:0.05',
    # '128x2_nc_r5_nopos_d0.05': 'encoder_hidden_sizes:128,128+decoder_hidden_sizes:128,128+context_feature:1+fullcontext_feature:0+position_feature:0+context_radii:5+dropout_p:0.05',
    # '64x3_nc_r5_nopos_d0.05': 'encoder_hidden_sizes:64,64,64+decoder_hidden_sizes:64,64,64+context_feature:1+fullcontext_feature:0+position_feature:0+context_radii:5+dropout_p:0.05',

    # 'enc_64x2_dec_64x5_nc_r5_nopos_d0.05': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:64,64,64,64,64+context_feature:1+fullcontext_feature:0+position_feature:0+context_radii:5+dropout_p:0.05',
    # 'enc_64x2_dec_128x2_nc_r5_nopos_d0.05': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:128,128+context_feature:1+fullcontext_feature:0+position_feature:0+context_radii:5+dropout_p:0.05',
    # 'enc_64x2_dec_256x2_nc_r5_nopos_d0.05': 'encoder_hidden_sizes:64,64+decoder_hidden_sizes:256,256+context_feature:1+fullcontext_feature:0+position_feature:0+context_radii:5+dropout_p:0.05',

  }

  num_scripts = 0
  for exp_nm in exp_nms:
    for train_test_id in range(3):
      for hyperparam_nm in hyperparam_combinations:
        hyperparam_setting = hyperparam_combinations[hyperparam_nm]
        run_nm = '%s_%s' % (run_prefix, hyperparam_nm)

        command = 'python %s.py %s %s %s %s' % (NAME, exp_nm, run_nm, train_test_id, hyperparam_setting)
        script_id = NAME.split('_')[0]

        # Write shell scripts
        sh_fn = qsubs_dir + 'q_%s_%s_%s_%s.sh' % (script_id, exp_nm, run_nm, train_test_id)
        with open(sh_fn, 'w') as f:
          f.write('#!/bin/bash\n%s\n' % (command))
        num_scripts += 1

        # Write qsub commands
        qsub_commands.append('qsub -V -l h_rt=16:00:00,h_vmem=3G -l os=RedHat7 -wd %s %s &' % (_config.SRC_DIR, sh_fn))

  # Save commands
  commands_fn = qsubs_dir + '_commands.sh'
  with open(commands_fn, 'w') as f:
    f.write('\n'.join(qsub_commands))

  subprocess.check_output('chmod +x %s' % (commands_fn), shell = True)

  print('Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir))
  return

if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(
      dataset_id = sys.argv[1]
    )
  elif len(sys.argv) == 4:
    main(
      dataset_id = sys.argv[1], 
      custom_fold_nm = sys.argv[2], 
      train_test_id = sys.argv[3], 
    )
  elif len(sys.argv) == 5:
    main(
      dataset_id = sys.argv[1], 
      custom_fold_nm = sys.argv[2], 
      train_test_id = sys.argv[3], 
      custom_hyperparams = sys.argv[4]
    )
  else:
    gen_qsubs()
    # main()
