import torch
from torch import FloatTensor as FT
from torch import LongTensor as LT
from torch import nn
from torch.nn import functional as F
import os
from collections import OrderedDict
import calendar
import time
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions.normal import Normal
import math
from pathlib import Path
from scipy.signal import argrelextrema

torch.manual_seed(0)
EPSILON = math.pow(10, -6)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print('Current device  >> ', DEVICE)
print('=========================== ')


def get_Activation(act):
    if act == 'tanh': return nn.Tanh()
    if act == 'sigmoid': return nn.Sigmoid()
    if act == 'none': return nn.Identity()
    return nn.ReLU()


class AE_encoder(nn.Module):
    def __init__(
            self,
            device,
            inp_dim,
            structure_config,
            dropout=0.05
    ):
        super(AE_encoder, self).__init__()
        self.device = device
        self.structure_config = structure_config

        encoder_FCN_to_latent_dims = structure_config['encoder_FCN_to_latent']['dims']
        _act = structure_config['encoder_FCN_to_latent']['activation']
        self.ae_latent_dimension = structure_config['ae_latent_dimension']

        _layers = []
        inp = inp_dim
        for k in encoder_FCN_to_latent_dims + [self.ae_latent_dimension]:
            _layers.append(nn.Linear(inp, k))
            _layers.append(nn.Dropout(dropout))
            _layers.append(get_Activation(_act))
            inp = k
        self.FC_z = nn.Sequential(*_layers)
        return

    def forward(self, X):
        op = self.FC_z(X)
        return op


class AE_decoder(nn.Module):
    def __init__(
            self,
            device,
            op_dim,
            structure_config=None,
            dropout=0.05
    ):
        super(AE_decoder, self).__init__()
        self.device = device
        self.structure_config = structure_config

        latent_dim = structure_config['ae_latent_dimension']
        inp_dim = latent_dim
        fcn_dims = structure_config['decoder_FC_from_latent']['dims']
        _act = structure_config['decoder_FC_from_latent']['activation']
        _layers = []

        for k in fcn_dims:
            _layers.append(nn.Linear(inp_dim, k))
            _layers.append(nn.Dropout(dropout))
            _layers.append(get_Activation(_act))
            inp_dim = k

        _layers.append(nn.Linear(inp_dim, op_dim))
        _layers.append(nn.Sigmoid())

        self.FC_z = nn.Sequential(*_layers)
        # the output size is inp_dim
        return

    def forward(self, z):
        z1 = self.FC_z(z)
        return z1


class AE(nn.Module):
    def __init__(
            self,
            device,
            data_dim,
            encoder_structure_config,
            decoder_structure_config,
            dropout = 0.05
    ):

        super(AE, self).__init__()
        inp_dim = data_dim
        op_dim = data_dim
        self.device = device
        self.encoder = AE_encoder(
            device,
            inp_dim,
            encoder_structure_config,
            dropout
        )
        self.decoder = AE_decoder(
            device,
            op_dim,
            decoder_structure_config,
            dropout
        )

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.mode = None
        return

    def forward(self, x):
        z = self.encoder(x)

        if self.mode == 'compress':
            return z
        x_recon = self.decoder(z)
        return x_recon, z


class AE_loss_module(nn.Module):
    def __init__(
            self,
            device,
            structure_config=None
    ):
        super(AE_loss_module, self).__init__()
        self.device = device
        self.structure_config = structure_config
        num_fields = len(structure_config['discrete_dims']) + 1
        # ===========
        # config Format :
        # decoder output dim , loss type , data type
        # decoder output dim is the number of categories for onehot data
        # ===========

        # self.loss_FC = nn.Linear(num_fields, 1, bias=False)
        print('Loss structure config', structure_config)
        self.discrete_dims = self.structure_config['discrete_dims']

        real_dims = self.structure_config['real_dims']
        split_schema = []
        for column, dim in self.discrete_dims.items():
            if dim == 2:  # Binary case
                dim = 1
            split_schema.append(dim)
        split_schema.append(real_dims)
        print(' Loss module split schema ', split_schema)
        self.split_schema = split_schema
        self.real_dims = real_dims
        return

    # =================================
    # Converting to simple MSE
    # =================================
    def forward(
            self,
            x_true,
            x_pred
    ):
        # ------------------
        # Split the x
        # ------------------
        x_true = x_true.to(self.device)
        x_pred = x_pred.to(self.device)

        mse = F.mse_loss(x_true, x_pred, reduction=None)
        return mse


class model_6_abl_1(nn.Module):
    def __init__(
            self,
            device,
            latent_dim,
            data_dim,
            encoder_structure_config,
            decoder_structure_config,
            loss_structure_config,
            ae_dropout,
            fc_dropout,
            include_noise = True
    ):
        super(model_6_abl_1, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.ae_module = AE(
            device=device,
            data_dim = data_dim,
            encoder_structure_config=encoder_structure_config,
            decoder_structure_config=decoder_structure_config,
            dropout=ae_dropout
        )
        self.ae_module = self.ae_module.to(self.device)
        self.ae_loss_module = AE_loss_module(self.device, loss_structure_config)
        self.ae_loss_module = self.ae_loss_module.to(self.device)

        self.num_fields = len(encoder_structure_config['discrete_dims']) + 1
        latent_dim = self.ae_module.encoder.ae_latent_dimension

        self.score_layer = nn.Sequential(
            nn.Linear( latent_dim, latent_dim // 2),
            nn.Dropout(fc_dropout),
            nn.Tanh(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
        self.score_layer.to(device)
        # Possible values : train, test
        self.include_noise = include_noise
        self.normal_noise_dist = Normal(
            loc=FT(np.zeros(latent_dim)),
            scale=FT(np.ones(latent_dim))
        )

        print(self.normal_noise_dist)
        self.mode = 'train'
        return

    def score_sample(self, z):
        x3 = self.score_layer(z)
        return x3

    def forward(
            self,
            x,
            sample_type='pos'
    ):
        global EPSILON
        # ================================================ #
        # Training mode
        # ================================================ #
        if self.mode == 'train':
            # Return the per sample loss
            x_recon, z = self.ae_module(x)
            mse_loss = self.ae_loss_module(
                x_true=x,
                x_pred=x_recon
            )
            # loss_md (multidimensional)  has shape [ batch, num_fields ]
            # Sample loss is sum of the loss pertaining to all the fields
            sample_loss = torch.sum(
                mse_loss,
                dim=-1,
                keepdim=True
            ).to(self.device)

            # ==========
            # If sample_type is neg:
            # reduce 1/loss ; i.e. increase loss for negative values
            if sample_type == 'neg':
                # sample_loss = torch.reciprocal(torch.log(sample_loss))
                batch_size = x.shape[0]
                if self.include_noise :
                    r_noise = self.normal_noise_dist.sample(sample_shape=[batch_size]).to(self.device)
                    z = z + r_noise

            sample_score = self.score_sample(z)
            return sample_loss, sample_score

        # ================================================ #
        elif self.mode == 'test':
            x_recon, z = self.ae_module(x)

            s = self.score_sample(z)
            return s


# ====================================================
# Main class
# ====================================================

class model_container():
    def __init__(
            self,
            data_set,
            device,
            latent_dim,
            data_dim,
            encoder_structure_config,
            decoder_structure_config,
            loss_structure_config,
            batch_size=256,
            learning_rate=0.05,
            max_gamma=5,
            log_interval=100,
            ae_dropout=0.05,
            fc_dropout=0.05,
            num_epochs =15,
            burn_in_epochs = 5,
            phase_2_epochs = 5,
            phase_3_epochs = 5,
            include_noise =True
    ):
        self.device = device
        self.data_set = data_set
        self.log_interval = log_interval
        self.LR = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_gamma = max_gamma

        self.t_stamp = calendar.timegm(time.gmtime())

        self.model_signature = '_'.join([self.data_set, str(self.t_stamp)])
        self.chkpt_folder = os.path.join('checkpoints', self.data_set, str(self.t_stamp))

        # Create the folder to place checkpoints in
        chkpt_path = Path(self.chkpt_folder)
        chkpt_path.mkdir(exist_ok=True, parents=True)
        print('Sacving checkpoints to :: ',  self.chkpt_folder)

        self.network_module = model_6_abl_1(
            device,
            latent_dim,
            data_dim,
            encoder_structure_config,
            decoder_structure_config,
            loss_structure_config,
            ae_dropout,
            fc_dropout,
            include_noise
        )
        self.burn_in_epochs = burn_in_epochs
        self.phase_2_epochs = phase_2_epochs
        self.phase_3_epochs = phase_3_epochs

        return

    def train_model(
            self,
            X_pos,
            X_neg=None
    ):
        self.network_module.ae_module.mode = 'train'
        self.network_module.ae_module.train()
        self.network_module.ae_loss_module.train()
        self.network_module.mode = 'train'
        learning_rate = self.LR

        parameters = list(self.network_module.parameters())
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate
        )

        log_interval = self.log_interval
        num_neg_samples = X_neg.shape[1]
        losses = []
        bs = self.batch_size

        burn_in_epochs = self.burn_in_epochs
        self.num_epochs = self.burn_in_epochs + self.phase_2_epochs + self.phase_3_epochs
        t_max = self.num_epochs
        t_start = burn_in_epochs
        last_phase = False
        epoch_phase = 0
        epoch_losses_phase_3 = []
        import pandas as pd

        df_phase_3_losses = pd.DataFrame(
            columns = [ 'epoch', 'loss' ]
        )

        for epoch in tqdm(range(1, self.num_epochs + 1)):
            t = epoch
            if epoch < burn_in_epochs:
                epoch_phase = 1

            elif burn_in_epochs < epoch <= self.burn_in_epochs + self.phase_2_epochs:
                epoch_phase = 2
            else:
                epoch_phase = 3

            if epoch_phase == 1 :
                lambda_1 = 1
                gamma = 1
                lambda_2 = 1
            elif epoch_phase == 2 :
                lambda_1 = np.exp(-t_start * (t - t_start) / t_start)
                lambda_2 = 1
            elif epoch_phase == 3:
                lambda_1 = 0.001
                lambda_2 = 1

            if epoch > burn_in_epochs:
                gamma = min(1 + np.exp((t - t_start) / (t_max - t_start) + 1), self.max_gamma)

            # At start of new phase reset optimizer
            if epoch == self.burn_in_epochs + self.phase_2_epochs:
                parameters = list(self.network_module.score_layer.parameters())
                self.optimizer = torch.optim.Adam(
                    parameters,
                    lr=learning_rate
                )

            epoch_losses = []
            num_batches = X_pos.shape[0] // bs + 1
            idx = np.arange(X_pos.shape[0])
            np.random.shuffle(idx)
            X_P = X_pos[idx]
            X_N = X_neg[idx]

            X_P = FT(X_P).to(self.device)
            X_N = FT(X_N).to(self.device)
            b_epoch_losses_phase_3 = []
            for b in range(num_batches):

                # self.network_module.zero_grad()
                self.optimizer.zero_grad()

                _x_p = X_P[b * bs: (b + 1) * bs]
                _x_n = X_N[b * bs: (b + 1) * bs]

                # Positive sample
                batch_loss_pos, sample_score_pos = self.network_module(_x_p, sample_type='pos')
                batch_loss_neg = []
                sample_scores_neg = []

                # Split _x_n into num_neg_samples parts along dim 1
                #  ns  * [ batch, 1, _ ]
                x_neg = torch.chunk(_x_n, num_neg_samples, dim=1)
                # for negative samples at index i

                for ns in x_neg:
                    ns = ns.squeeze(1)
                    n_sample_loss, n_sample_score = self.network_module(
                        ns,
                        sample_type='neg'
                    )

                    sample_scores_neg.append(n_sample_score)

                # Shape : [ batch, num_neg_samples ]
                sample_scores_neg = torch.cat(sample_scores_neg, dim=1)
                sample_scores_neg = sample_scores_neg.squeeze(1)

                # ========================
                # Loss 2 should be the scoring function
                # sample_score is of value between 0 and 1
                # Since we model last layer as logistic reg
                # ========================

                data_size = _x_p.shape[0]
                num_neg_samples = sample_scores_neg.shape[1]

                _scores = torch.cat([sample_score_pos, sample_scores_neg], dim=1)

                targets = torch.cat([torch.ones([data_size, 1]), torch.zeros([data_size, num_neg_samples])], dim=-1).to(
                    self.device)

                loss_2 = F.binary_cross_entropy(_scores, targets, reduction='none')
                loss_2_1 = gamma * loss_2[:, 0]  # positive
                loss_2_0 = loss_2[:, 1:]  # negatives
                loss_2 = loss_2_1 + torch.mean(loss_2_0, dim=1, keepdim=False)
                loss_2 = torch.mean(loss_2, dim=0, keepdims=False)

                # Standard AE loss

                loss_1 = torch.mean(batch_loss_pos, dim=0, keepdim=False)

                # batch_loss_neg = torch.clamp(batch_loss_neg, 0.0001,1)
                # loss_3 = torch.sum(batch_loss_neg, dim=1, keepdim=False)
                # loss_3 = torch.mean(loss_3, dim=0, keepdim=False)
                score_loss = lambda_2 * loss_2

                if epoch_phase == 1:
                    batch_loss = lambda_1 * loss_1

                elif epoch_phase == 2:
                    batch_loss = lambda_1 * loss_1
                    if b % 2 == 0:
                        batch_loss = batch_loss + score_loss
                elif epoch_phase == 3:
                    batch_loss = score_loss

                # ------------------------------------------
                # Record the estimator loss in last phase
                # -------------------------------------------
                if epoch_phase == 3:
                    b_epoch_losses_phase_3.append(score_loss.clone().cpu().data.numpy())
                # ====================
                # Clip Gradient
                # ====================

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network_module.parameters(), 2)
                self.optimizer.step()

                loss_value = batch_loss.clone().cpu().data.numpy()
                losses.append(loss_value)
                if b % log_interval == 0:
                    print(' Epoch {} Batch {} Loss {:.4f} || AE {:.4f} {:.4f} '.format(
                        epoch,
                        b,
                        batch_loss,
                        loss_1,
                        loss_2
                    )
                    )

                epoch_losses.append(loss_value)

            mean_epoch_loss =  np.mean(epoch_losses)
            print('Epoch loss ::',mean_epoch_loss)

            # ------------------
            # Save checkpoint
            # ------------------
            if epoch_phase == 3:
                epoch_losses_phase_3.append(np.mean(b_epoch_losses_phase_3))
                _path = os.path.join(self.chkpt_folder, 'epoch_{}'.format(epoch))
                torch.save(self.network_module.state_dict(), _path)
                df_phase_3_losses = df_phase_3_losses.append({
                    'epoch': epoch,
                    'loss' : np.mean(b_epoch_losses_phase_3)
                    }, ignore_index=True
                )

        # ===========
        # Find epoch with lowest loss
        # ===========
        best_epoch = self.find_lowest_loss_epoch(df_phase_3_losses)
        _path = os.path.join(self.chkpt_folder, 'epoch_{}'.format(int(best_epoch)))

        self.network_module.load_state_dict(torch.load(_path))
        self.network_module.mode = 'test'
        return losses, epoch_losses_phase_3

    # ===========
    # Find epoch with lowest loss
    # ===========
    def find_lowest_loss_epoch(self, df):
        epoch_num = list(df['epoch'])
        losses = list(df['loss'])

        # Case linearly decreasing loss
        if np.min(losses) == losses[-1]:
            return epoch_num[-1]

        # ----
        # Loss increases after reducing for at least 3 epochs
        # Find such a point in the second half
        # ----
        x = np.array(losses)
        idx_list = argrelextrema(x, np.less)[0][::-1]
        print('[DEBUG]', [(i,j) for i,j in zip(epoch_num, losses)])
        for idx in idx_list:
            if x[idx-1] > x[idx] and x[idx-2] > x[idx] and x[idx-3] > x[idx]:
                print( x[idx-3:idx+1])
                target_epoch = epoch_num[idx]
                print('Chosen index : {}; epoch -> {}'.format(idx,target_epoch))
                return int(target_epoch)

        # If all else fails ; return the last one
        return epoch_num[-1]

    def get_compressed_embedding(
            self,
            data
    ):
        self.network_module.eval()
        self.network_module.mode = 'test'
        self.network_module.ae_module.mode = 'compress'
        X = FT(data).to(self.device)
        bs = 500
        num_batches = data.shape[0] // self.batch_size + 1
        output = []
        for b in range(num_batches):
            _x = X[b * bs: (b + 1) * bs]
            z = self.network_module.ae_module(_x)
            z_data = z.clone().cpu().data.numpy()
            output.extend(z_data)
        return output

    # ================================================ #
    # Score the samples
    # ================================================ #
    def get_score(
            self,
            data
    ):
        self.network_module.eval()
        self.network_module.mode = 'test'
        self.network_module.ae_module.mode = 'test'
        X = FT(data).to(self.device)
        bs = 500
        num_batches = data.shape[0] // self.batch_size + 1
        output = []
        for b in range(num_batches):
            _x = X[b * bs: (b + 1) * bs]
            if _x.shape[0] == 0: continue
            z = self.network_module(_x)
            z_data = z.clone().cpu().data.numpy()
            output.extend(z_data)
        return output
