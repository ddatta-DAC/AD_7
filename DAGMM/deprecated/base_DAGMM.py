import torch
from torch import FloatTensor as FT
from torch import LongTensor as LT
from torch import nn
from torch.nn import functional as F
import os
import math
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print('Current device  >> ', DEVICE)
print('=========================== ')


# ===================
# For kddcup data
# FC(120, 60, tanh)
# FC(60, 30, tanh)
# FC(30, 10, tanh)
# FC(10, 1, none)
# FC(1, 10, tanh)
# FC(10, 30, tanh)
# FC(30, 60, tanh)
# FC(60, 120, none)
# Reconstruction features : Relative Euclidean distance
# Cosine similarity
# ====================

class AE_encoder(nn.Module):
    def __init__(
            self,
            device,
            structure_config
    ):

        super(AE_encoder, self).__init__()
        self.device = device
        self.structure_config = structure_config
        activation = structure_config['encoder_layers']['activation']
        layer_dims = structure_config['encoder_layers']['layer_dims']

        # ================
        # Concatenate the input
        # ================

        self.discrete_column_dims = structure_config['discrete_column_dims']
        self.num_discrete_columns = structure_config['num_discrete']
        self.num_real_columns = structure_config['num_real']

        input_dim = sum(list(self.discrete_column_dims.values())) + self.num_real_columns
        layers = []
        # Triplet <  output_dim, activation >
        num_layers = len(layer_dims)
        for idx in range(num_layers):
            op_dim = layer_dims[idx]
            layers.append(nn.Linear(
                input_dim, op_dim
            ))
            if idx == num_layers - 1: activation = 'none'
            if activation == 'tanh':
                layers.append(
                    nn.Tanh()
                )
            elif activation == 'sigmoid':
                layers.append(
                    nn.Sigmoid()
                )
            elif activation == 'none':
                pass

            input_dim = op_dim

        self.FC_z = nn.Sequential(*layers)
        return

    def forward(self, X):

        real_x_0 = X[:, -self.num_real_columns:].type(torch.FloatTensor).to(self.device)
        discrete_x_0 = X[:, :self.num_discrete_columns].type(torch.LongTensor).to(self.device)
        discrete_x_1 = torch.chunk(discrete_x_0, self.num_discrete_columns, dim=1)
        # discrete_x_1 is an array
        res = []
        column_name_list = list(self.discrete_column_dims.keys())
        for idx in range(self.num_discrete_columns):
            col = column_name_list[idx]
            _x = discrete_x_1[idx].to(self.device)
            n_cat = self.discrete_column_dims[col]
            _x = F.one_hot(_x, n_cat).type(FT).squeeze(1).to(self.device)
            res.append(_x)

        res.append(real_x_0)
        x_concat = torch.cat(res, dim=1)

        # ==============
        # Conactenated input : res
        # ==============
        op = self.FC_z(x_concat)
        return op


class AE_decoder(nn.Module):
    def __init__(
            self,
            device,
            structure_config=None,
    ):
        super(AE_decoder, self).__init__()
        self.device = device
        self.structure_config = structure_config
        self.discrete_column_dims = structure_config['discrete_column_dims']
        self.num_discrete_columns = structure_config['num_discrete']
        self.num_real_columns = structure_config['num_real']
        final_output_dim = structure_config['final_output_dim']
        activation = structure_config['decoder_layers']['activation']
        layer_dims = structure_config['decoder_layers']['layer_dims']

        layers = []
        inp_dim = layer_dims[0]
        num_layers = len(layer_dims)
        for idx in range(1, num_layers):
            op_dim = layer_dims[idx]
            layers.append(nn.Linear(inp_dim, op_dim))
            if activation == 'tanh':
                layers.append(
                    nn.Tanh()
                )
            elif activation == 'sigmoid':
                layers.append(
                    nn.Sigmoid()
                )
            elif activation == 'none':
                layers.append(
                    nn.Sigmoid()
                )
            inp_dim = op_dim

        self.FC_z = nn.Sequential(*layers)
        return

    def forward(self, z):
        res = self.FC_z(z)
        return res


class AE(nn.Module):

    def __init__(
            self,
            device,
            encoder_structure_config,
            decoder_structure_config
    ):
        super(AE, self).__init__()
        self.device = device
        self.encoder = AE_encoder(
            device,
            encoder_structure_config
        )
        self.decoder = AE_decoder(
            device,
            decoder_structure_config
        )

        self.encoder.to(self.device)
        self.decoder.to(self.device)
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
        # ===========
        # config Format :
        # decoder output dim , loss type , data type
        # decoder output dim is the number of categories for onehot data
        # ===========

        self.structure_config = structure_config
        print('Loss structure config', structure_config)
        return

    def forward(
            self,
            x_true,
            x_pred,
            mode='opt'
    ):
        # ------------------
        # If mode is optimization, do a mean across the batch dimension
        # ------------------

        split_schema_true = []

        # structure_config is a list of dictionaries with keys: dim, type

        for _ in self.structure_config:
            if _['type'] == 'onehot':
                split_schema_true.append(1)
            else:
                split_schema_true.append(_['dim'])

        _x_true = torch.split(
            x_true,
            split_schema_true,
            dim=1
        )
        res = []
        for idx in range(len(_x_true)):
            if self.structure_config[idx]['type'] == 'onehot':
                num_cat = self.structure_config[idx]['dim']
                _x_ = F.one_hot(
                    _x_true[idx].type(torch.LongTensor),
                    num_cat
                ).squeeze(1).type(torch.FloatTensor)
                res.append(_x_)
            else:
                res.append( _x_true[idx])
        res = torch.cat(res, dim=1)
        res = res.to(self.device)
        x_true = res
        # =======================
        # Expand the true x for the discrete attributes to one hot
        # =======================
        x_true = x_true.to(self.device)
        x_pred = x_pred.to(self.device)

        loss_1 = F.mse_loss(
            x_pred,
            x_true,
            reduction='none'
        )
        loss_1 = torch.sum(loss_1, dim=1, keepdim=True).to(self.device)
        l2_norm = torch.norm(x_true, keepdim=True, dim=1).to(self.device)
        loss_1 = loss_1 / l2_norm
        loss_1 = loss_1.to(self.device)
        loss_2 = 1 - F.cosine_similarity(x_pred, x_true).to(self.device)
        loss_2 = loss_2.unsqueeze(1)

        if mode == 'opt':
            loss = loss_1 + loss_2
            loss = torch.sum(loss, dim=1, keepdim=False)
            loss_scalar = torch.mean(loss, dim=0, keepdim=False)
            loss_per_sample = torch.cat([loss_1, loss_2], dim=1)
            return loss_scalar, loss_per_sample

        else:
            loss = torch.cat([loss_1, loss_2], dim=1)
            return loss


# ===========================================
# Estimator network
# ===========================================

class GMM(nn.Module):

    def __init__(
            self,
            device,
            structure_config
    ):
        super(GMM, self).__init__()

        FCN_dims = structure_config['FC_layer_dims']
        num_components = structure_config['num_components']
        activation = structure_config['FC_activation']
        FC_dropout = structure_config['FC_dropout']

        self.device = device
        self.K = num_components
        self.input_dim = FCN_dims[0]
        inp_dim = FCN_dims[0]

        layers = []
        num_layers = len(FCN_dims)
        for i in range(1, num_layers):
            op_dim = FCN_dims[i]
            layers.append(nn.Linear(inp_dim, op_dim))
            if i != num_layers - 1:
                layers.append(nn.Dropout(FC_dropout))

            if activation == 'tanh' and i != num_layers - 1:
                layers.append(nn.Tanh())
            elif activation == 'sigmoid' and i != num_layers - 1:
                layers.append(nn.Sigmoid())
            inp_dim = op_dim

        self.FCN_1 = nn.Sequential(*layers)
        print('GMM ::', self.FCN_1)
        return

    def forward(self, z):

        member_vec = self.FCN_1(z)
        # Shape [ batch, K]
        softmax_member_vec = F.softmax(
            member_vec,
            dim=1
        )

        # -----------
        # Estimate mixture ratios (priors)
        # Shape : [K] ; no batch dim
        theta = torch.mean(
            softmax_member_vec,
            dim=0,
            keepdim=False
        )
        theta = theta.squeeze(0)
        # print('Theta shape', theta.shape)

        # Mean of the components
        K = self.K

        # -------------
        MU = []
        VAR = []
        for k in range(K):
            gamma = softmax_member_vec[:, k]  # Shape [Batch,1]
            sum_gamma = torch.sum(gamma, dim=0, keepdim=False)  # Shape [1]
            num = gamma.unsqueeze(1) * z
            num = torch.sum(num, dim=0, keepdim=False)  # Shape[ latent_dim ]
            denom = sum_gamma
            mu_k = num / denom  # Shape [ latent_dim ]

            num = gamma.unsqueeze(1) * torch.pow((z - mu_k), 2)
            num = torch.sum(num, dim=0, keepdim=False)
            denom = sum_gamma
            var_k = num / denom
            MU.append(mu_k)
            VAR.append(var_k)

        mean_vectors = torch.stack(MU, dim=0)
        var_vectors = torch.stack(VAR, dim=0)
        # print('Shape of mu ', mean_vectors.shape)
        # print('Shape of var_vectors ', var_vectors.shape)

        return theta, mean_vectors, var_vectors


# =============================================

class GMM_loss_module(nn.Module):
    def __init__(self):
        super(GMM_loss_module, self).__init__()
        return

    def forward(self, z, theta, mu, sigma):
        num_components = list(theta.shape)[-1]
        # print('Number of gmm components', num_components)

        # list_theta = torch.split(theta, num_components)
        # list_mu = torch.chunk(mu, num_components, dim=0)
        # list_sigma = torch.chunk(sigma, num_components, dim=0)
        batch_dim = list(z.shape)[0]
        latent_dim = list(z.shape)[1]

        # Calculate the sample energy per sample
        P_k = []
        for k in range(num_components):
            theta_k = theta[k]
            mu_k = mu[k]
            sigma_k = sigma[k]  # this actually sigma squared in the equations !!
            a = torch.pow(z - mu_k.repeat(batch_dim, 1), 2)  # shape  [batch, latent_dim]
            a = torch.sum(a, dim=1, keepdim=True)  # shape  [batch, 1]
            b = torch.prod(sigma_k, dim=0).repeat(batch_dim, 1)  # shape  [batch, 1]
            c = torch.exp(-0.5 * a / b)  # shape  [batch, 1]

            # Denominator
            _d = torch.prod(sigma_k, dim=0)
            _2_pi = math.pi * 2
            d = torch.reciprocal(np.power( _2_pi, latent_dim / 2) * torch.sqrt(_d))
            P = theta_k * d * c  # shape [batch,1]
            P_k.append(P)

        # Shape [batch, num_components]
        P_k = torch.stack(
            P_k,
            dim=1
        )

        # Shape [batch, 1]
        P = -torch.log(
            torch.sum(
                P_k,
                dim=1,
                keepdim=False
            )
            + 1 / np.power(10, 5)
        )
        # Shape [Batch,]
        P = P.squeeze(1)
        # Scalar
        E_z_batch = torch.mean(P, dim=0, keepdim=False)
        return E_z_batch

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        # Compute the energy based on the specified gmm params.
        # If none are specified use the cached values.

        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)
        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D) * eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append(np.linalg.det(cov_k.data.cpu().numpy() * (2 * np.pi)))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag
# ====================================================
# Main class
# ====================================================

class DAGMM_base_model():

    def __init__(
            self,
            device,
            latent_dim,
            encoder_structure_config,
            decoder_structure_config,
            gmm_structure_config,
            loss_structure_config,
            optimizer='Adam',
            batch_size=256,
            num_epochs=10,
            learning_rate=0.05,
            l2_regularizer=0.01
    ):
        self.device = device
        self.ae_obj = AE(
            device=device,
            encoder_structure_config=encoder_structure_config,
            decoder_structure_config=decoder_structure_config
        )

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = self.ae_obj.to(self.device)
        self.ae_loss_module = AE_loss_module(self.device,loss_structure_config)
        self.ae_loss_module = self.ae_loss_module.to(self.device)

        self.gmm_module = GMM(
            device=self.device,
            structure_config=gmm_structure_config
        )
        self.gmm_module = self.gmm_module.to(self.device)
        self.gmm_loss_module = GMM_loss_module()
        self.gmm_loss_module = self.gmm_loss_module.to(self.device)
        parameters_list = list(self.ae_obj.parameters()) + list(self.gmm_module.parameters())

        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(parameters_list, lr=learning_rate)
        else:
            self.optimizer = torch.optim.Adagrad(
                parameters_list,
                lr=learning_rate,
                weight_decay=l2_regularizer
            )
        return

    def train_model(
            self,
            data
    ):
        self.ae_obj.mode = 'train'
        self.ae_obj.train()
        log_interval = 100

        losses = []
        bs = self.batch_size
        lambda_1 = 0.1
        lambda_2 = 0.005

        for epoch in tqdm(range(self.num_epochs)):
            num_batches = data.shape[0] // bs + 1
            epoch_losses = []
            np.random.shuffle(data)
            X = FT(data).to(self.device)
            for b in range(num_batches):
                self.optimizer.zero_grad()
                _x = X[b * bs: (b + 1) * bs]
                x_recon, z = self.ae_obj(_x)

                ae_loss_scalar, ae_loss_per_sample = self.ae_loss_module(
                    x_true=_x,
                    x_pred=x_recon
                )

                gmm_input = torch.cat([z,ae_loss_per_sample], dim=1 )

                theta, mean_vectors, var_vectors = self.gmm_module(gmm_input)
                batch_loss_gmm = self.gmm_loss_module(
                    z, theta, mean_vectors, var_vectors
                )

                batch_loss_gmm = lambda_1 * batch_loss_gmm
                # Update weight

                batch_loss = ae_loss_scalar + batch_loss_gmm
                batch_loss.backward()

                # ====================
                # Clip Gradient
                # ====================
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                loss_value = batch_loss.clone().cpu().data.numpy()

                losses.append(loss_value)
                self.optimizer.step()

                # ====
                loss_value_gmm = np.mean(batch_loss_gmm.clone().cpu().data.numpy())
                loss_value_ae = ae_loss_scalar.clone().cpu().data.numpy()

                if b % log_interval == 0:
                    print(' Epoch {} Batch {} Loss {:.4f} [ AE {:.4f} GMM {:.4f}'.format(
                        epoch,
                        b,
                        loss_value,
                        loss_value_ae,
                        loss_value_gmm
                    )
                    )
                epoch_losses.append(loss_value)
            print('Epoch loss ::', np.mean(epoch_losses))
        return losses

    def get_compressed_embedding(
            self,
            data
    ):
        self.ae_obj.eval()
        self.ae_obj.mode = 'compress'
        X = FT(data).to(self.device)
        bs = self.batch_size
        num_batches = data.shape[0] // self.batch_size + 1
        output = []
        for b in range(num_batches):
            _x = X[b * bs: (b + 1) * bs]
            z = self.ae_obj(_x)
            z_data = z.clone().cpu().data.numpy()
            output.extend(z_data)

        return output

    def get_sample_energy(self, X):
        return
