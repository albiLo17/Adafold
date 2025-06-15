import copy
from Adafold.model.utils import chamfer
import torch
import torch.nn as nn
from .base_models import AutoEncoder, PointNet2Cls, PointNet2Seg
import torch.optim.lr_scheduler as lr_scheduler
import random

class RMA_MB(nn.Module):

    def __init__(self, args, device):
        super(RMA_MB, self).__init__()
        self.conditioning = args.dyn_conditioning
        self.inv_dyn = args.inv_dyn
        self.pretrained_enc = args.pretrained_enc

        self.rw = args.rw

        self.PI_encoder = AutoEncoder(args)
        self.adaptation_model = PointNet2Cls(args)
        self.fwd_model = PointNet2Seg(args)

        # prediction horizon
        self.H = args.H

        # Training details
        self.device = device
        if args.l2_reg:
            self.opt = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=1e-5)
        else:
            self.opt = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=0)#1e-5)
        self.lr_schedule = args.lr_schedule
        if self.lr_schedule:
            total_sched_epochs = 50
            final_lr = args.lr
            lr = final_lr*10
            self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
            self.scheduler_lambda = lr_scheduler.LambdaLR(self.opt,
                                                          lr_lambda=lambda epoch: lr * (final_lr / lr) ** (epoch / (total_sched_epochs - 1)) if epoch < total_sched_epochs else lr * (final_lr / lr))


        self.RMA_scheduler = args.RMA_schedule
        self.epsilon = 1.
        self.decay_rate = 0.985


        self.alpha = args.alpha     # prediction loss coefficient
        self.beta = args.beta       # adaptation loss coefficient
        self.mse = nn.MSELoss()
        self.loss_type = args.loss
        if self.loss_type == 'MSE':
            self.loss = nn.MSELoss()

        if 'chamfer' in self.loss_type:
            self.loss = chamfer
            self.bidirectional = False
            if 'bi' in self.loss_type:
                self.bidirectional = True

        if self.loss_type == 'MAE':
            self.loss = nn.L1Loss(reduction='mean')

    def forward(self, x, z=None, get_adapt=True):
        """
        If z is provided, it is going to be used to for the forward model.
        Otherwise, it will be obtained from the PI in case the model requires it
        :param x: batch=[batch_probe, batch_forward, params]
        :param z: context
        :param get_adapt: compute the adapted z
        :return: future_pcd_pred, pi_rec, z, z_adapt
        """""

        pi_rec = None
        if z is None:
            z, pi_rec = self.get_z(pi=x[2], batch_probe=x[0])

        z_adapt = None
        if get_adapt and self.conditioning == 2:       
            z_adapt = self.get_adapt(x[0])

        future_pcd_pred = self.fwd_model(x[1], encoding=z)

        return future_pcd_pred, pi_rec, z, z_adapt

    def get_adapt(self, x):
        return self.adaptation_model(x)

    def get_z(self, pi, batch_probe, test_phase=False):
        # NC baseline
        if self.conditioning == 0:
            return None, None
        # PI baseline
        if self.conditioning == 1:
            return pi, None
        # RMA
        if self.conditioning == 2:
            if not test_phase and not self.rw:      # if rw then we don't have PI, we need to leverage the adaptation model
                if not self.RMA_scheduler:
                    self.sampled_PI = True
                    return self.PI_encoder(pi)

                # with increasing probability, sample from adaptation model
                self.sampled_PI = random.uniform(0, 1) < self.epsilon
                if self.sampled_PI:
                    self.epsilon *= self.decay_rate
                    return self.PI_encoder(pi)
                else:
                    self.epsilon *= self.decay_rate
                    return self.adaptation_model(batch_probe), None
            else:
                self.sampled_PI = False
                return self.adaptation_model(batch_probe), None
            
        # RECURRENT
        if self.conditioning == 3:
            return self.adaptation_model(batch_probe), None

    def train_epoch(self, dataloader):
        self.train()
        batch_loss = 0
        batch_dyn_loss = 0
        batch_adapt_loss = 0

        for ix, batch in enumerate(dataloader):
            # [batch_z, batch_forward, params]
            batch[0] = batch[0].to(self.device)
            batch[1] = [b.to(self.device) for b in batch[1]]
            batch[2] = batch[2].to(self.device)

            # make self.H forward steps prediction where self.H is the prediction horizon
            for i in range(self.H):
                if i == 0:
                    future_pcd_pred, pi_re, z, z_adapt = self.forward([batch[0],  batch[1][i], batch[2]])
                else:
                    # update batch[1] with the predicted pointcloud
                    batch[1][i].x[:, :3] = future_pcd_pred
                    batch[1][i].pos = future_pcd_pred

                    future_pcd_pred, pi_re, _, _ = self.forward([batch[0],  batch[1][i], batch[2]], z=z_adapt, get_adapt=False)


                if 'chamfer' in self.loss_type:
                    loss_dyn = 0
                    num_batches = batch[1].batch.max() + 1
                    for b in range(num_batches):
                        loss_dyn += self.loss(future_pcd_pred[batch[1][i].batch == b], batch[3][i].x[batch[3][i].batch == b], self.bidirectional)
                        # loss_dyn += self.loss(future_pcd_pred[batch[1].batch == b], batch[3][i].pos[batch[3].batch == b], self.bidirectional)
                    loss_dyn /= num_batches
                else:
                    loss_dyn = self.loss(future_pcd_pred, batch[1][i].y)

                loss_adapt = 0

                if self.conditioning == 2 and self.sampled_PI and i == 0:
                    loss_adapt = self.beta * self.epsilon * self.mse(z_adapt, z.detach())

                loss = loss_dyn + loss_adapt

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            batch_loss += loss.item()  # .cpu().item()
            batch_dyn_loss += loss_dyn.item()  # .cpu().item()

            if hasattr(loss_adapt, 'detach'):
                batch_adapt_loss += loss_adapt.item()  # .cpu().item()
            else:
                batch_adapt_loss += loss_adapt

        batch_loss /= len(dataloader)
        batch_dyn_loss /= len(dataloader)
        batch_adapt_loss /= len(dataloader)

        loss_info = {'Train/Loss': batch_loss,
                     'Train/Loss Dyn': batch_dyn_loss,
                     'Train/Loss Adapt': batch_adapt_loss,
                     }

        image_info = {}
        if self.inv_dyn == 0:
            image_info = {'pred': future_pcd_pred[batch[1][-1].batch == batch[1][-1].batch.max()],
                          'label': batch[1][-1].y[batch[1][-1].batch == batch[1][-1].batch.max()]}

        if self.lr_schedule:
            self.scheduler_lambda.step()

        return loss_info, image_info


    @torch.no_grad()
    def val_epoch(self, dataloader):
        self.eval()
        batch_loss = 0
        batch_dyn_loss = 0
        batch_dyn_adapt_loss = 0
        batch_adapt_loss = 0

        with torch.no_grad():
            for ix, batch in enumerate(dataloader):
                # [batch_z, batch_forward, params]
                batch[0] = batch[0].to(self.device)
                batch[1] = [b.to(self.device) for b in batch[1]]
                batch[2] = batch[2].to(self.device)

                # make self.H forward steps prediction where self.H is the prediction horizon
                for i in range(self.H):
                    if i == 0:
                        future_pcd_pred, pi_re, z, z_adapt = self.forward([batch[0], batch[1][i], batch[2]])
                    else:
                        # update batch[1] with the predicted pointcloud
                        batch[1][i].x[:, :3] = future_pcd_pred
                        batch[1][i].pos = future_pcd_pred

                        future_pcd_pred, pi_re, _, _ = self.forward([batch[0], batch[1][i], batch[2]], z=z_adapt,
                                                                    get_adapt=False)

                    if 'chamfer' in self.loss_type:
                        loss_dyn = 0
                        num_batches = batch[1].batch.max() + 1
                        for b in range(num_batches):
                            loss_dyn += self.loss(future_pcd_pred[batch[1][i].batch == b],
                                                  batch[3][i].x[batch[3][i].batch == b], self.bidirectional)
                        loss_dyn /= num_batches
                    else:
                        loss_dyn = self.loss(future_pcd_pred, batch[1][i].y)

                    loss_adapt = 0

                    if self.conditioning == 2 and self.sampled_PI and i == 0:
                        loss_adapt = self.beta * self.epsilon * self.mse(z_adapt, z.detach())

                    loss = loss_dyn + loss_adapt

                batch_loss += loss.item()  # .cpu().item()
                batch_dyn_loss += loss_dyn.item()  # .cpu().item()

                # this loss does not have detach if args.dyn_conditioning = 0 or 1
                if hasattr(loss_adapt, 'detach'):
                    batch_adapt_loss += loss_adapt.item()  # .cpu().item()
                else:
                    batch_adapt_loss += loss_adapt

            batch_loss /= len(dataloader)
            batch_dyn_loss /= len(dataloader)
            batch_dyn_adapt_loss /= len(dataloader)
            batch_adapt_loss /= len(dataloader)

            loss_info = {'Val/Loss': batch_loss,
                         'Val/Loss Dyn': batch_dyn_loss,
                         'Val/Loss Adapt': batch_adapt_loss,
                         }

            image_info = {}
            if self.inv_dyn == 0:
                image_info = {'pred': future_pcd_pred[batch[1][-1].batch == batch[1][-1].batch.max()],
                              'label': batch[1][-1].y[batch[1][-1].batch == batch[1][-1].batch.max()]}

        return loss_info, image_info


    def load_dict(self, model_path=None, encoder_paths=None, load_full=True, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # load full
        if model_path is not None:
            state_dict = torch.load(model_path, map_location=torch.device(device))
            if load_full:
                self.load_state_dict(state_dict)
                print('Successfully loaded full model.')
            else:
                key_adapt = [key for key in state_dict.keys() if 'adaptation' in key]
                adapt_dict_subset = {key: state_dict[key] for key in key_adapt}

                key_PI = [key for key in state_dict.keys() if 'PI' in key]
                PI_dict_subset = {key: state_dict[key] for key in key_PI}

                self.load_state_dict(adapt_dict_subset, strict=False)
                self.load_state_dict(PI_dict_subset, strict=False)
                print('Successfully loaded encoders.')

        if encoder_paths is not None:
            model_keys = self.state_dict().keys()

            state_dict_adapt = torch.load(encoder_paths[1], map_location=torch.device(device))
            state_dict_PI = torch.load(encoder_paths[0], map_location=torch.device(device))

            adapt_dict_subset = {f'adaptation_model.{key}': state_dict_adapt[key] for key in state_dict_adapt if
                                 f'adaptation_model.{key}' in model_keys}

            PI_dict_subset = {f'PI_encoder.{key}': state_dict_PI[key] for key in state_dict_PI if
                              f'PI_encoder.{key}' in model_keys}

            self.load_state_dict(adapt_dict_subset, strict=False)
            self.load_state_dict(PI_dict_subset, strict=False)
            print('Successfully loaded encoders.')

    def freeze_params(self, all=False):
        # to see the parameters list(self.PI_encoder.parameters())
        if all:
            for param in self.parameters():
                param.requires_grad = False
            print('All parameters freezed.')
        else:
            # Freeze only encoders
            for param in self.PI_encoder.parameters():
                param.requires_grad = False
            for param in self.adaptation_model.parameters():
                param.requires_grad = False

            print('Encoders parameters freezed.')