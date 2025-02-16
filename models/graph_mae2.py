import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module, Linear, PReLU, Parameter, Sequential
from typing import Optional

from .gat import GAT

def setup_module(enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True):
    return GAT(
        in_dim=in_dim,
        num_hidden=num_hidden,
        out_dim=out_dim,
        num_layers=num_layers,
        nhead=nhead,
        nhead_out=nhead_out,
        concat_out=concat_out,
        activation=activation,
        feat_drop=dropout,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        norm=norm,
        encoding=(enc_dec == "encoding"),
    )

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


class GraphMAE2(Module):
    '''
    GraphMAE2
    https://arxiv.org/abs/2304.04779
    Self-supervised Learning
    Based on: https://github.com/THUDM/GraphMAE2/blob/main/models/edcoder.py
    '''
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            num_dec_layers: int,
            num_remasking: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            remask_rate: float = 0.5,
            remask_method: str = "random",
            mask_method: str = "random",
            loss_fn: str = "byol",
            drop_edge_rate: float = 0.0,
            alpha_l: float = 2,
            lam: float = 1.0,
            delayed_ema_epoch: int = 0,
            momentum: float = 0.996,
            replace_rate: float = 0.0,
            zero_init: bool = False,
         ):
        super().__init__()
        self._mask_rate = mask_rate
        self._remask_rate = remask_rate
        self._mask_method = mask_method
        self._alpha_l = alpha_l
        self._delayed_ema_epoch = delayed_ema_epoch

        self.num_remasking = num_remasking
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._momentum = momentum
        self._replace_rate = replace_rate
        self._num_remasking = num_remasking
        self._remask_method = remask_method

        self._token_rate = 1 - self._replace_rate
        self._lam = lam

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0

        enc_num_hidden = num_hidden // nhead
        enc_nhead = nhead

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead 

        # build encoder
        self.encoder = setup_module(
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.decoder = setup_module(
                enc_dec="decoding",
                in_dim=dec_in_dim,
                num_hidden=dec_num_hidden,
                out_dim=in_dim,
                nhead_out=nhead_out,
                num_layers=num_dec_layers,
                nhead=nhead,
                activation=activation,
                dropout=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm,
                concat_out=True,
            )

        self.enc_mask_token = Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = Parameter(torch.zeros(1, num_hidden))

        self.encoder_to_decoder = Linear(dec_in_dim, dec_in_dim, bias=False)
        
        if not zero_init:
            self.reset_parameters_for_token()


        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        
        self.projector = Sequential(
            Linear(num_hidden, 256),
            PReLU(),
            Linear(256, num_hidden),
        )
        self.projector_ema = Sequential(
            Linear(num_hidden, 256),
            PReLU(),
            Linear(256, num_hidden),
        )
        self.predictor = Sequential(
            PReLU(),
            Linear(num_hidden, num_hidden)
        )
        
        self.encoder_ema = setup_module(
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.projector_ema.load_state_dict(self.projector.state_dict())

        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()



    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            print(f"=== Use mse_loss ===")
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            print(f"=== Use sce_loss and alpha_l={alpha_l} ===")
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, g, x, targets=None, epoch=0, drop_g1=None, drop_g2=None):        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x, targets, epoch, drop_g1, drop_g2)

        return loss

    def mask_attr_prediction(self, g, x, targets, epoch, drop_g1=None, drop_g2=None):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        use_g = drop_g1 if drop_g1 is not None else g

        enc_rep = self.encoder(use_g, use_x,)

        with torch.no_grad():
            drop_g2 = drop_g2 if drop_g2 is not None else g
            latent_target = self.encoder_ema(drop_g2, x,)
            if targets is not None:
                latent_target = self.projector_ema(latent_target[targets])
            else:
                latent_target = self.projector_ema(latent_target[keep_nodes])

        if targets is not None:
            latent_pred = self.projector(enc_rep[targets])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)
        else:
            latent_pred = self.projector(enc_rep[keep_nodes])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)

        # ---- attribute reconstruction ----
        origin_rep = self.encoder_to_decoder(enc_rep)

        loss_rec_all = 0
        if self._remask_method == "random":
            for i in range(self._num_remasking):
                rep = origin_rep.clone()
                rep, remask_nodes, rekeep_nodes = self.random_remask(use_g, rep, self._remask_rate)
                recon = self.decoder(pre_use_g, rep)

                x_init = x[mask_nodes]
                x_rec = recon[mask_nodes]
                loss_rec = self.criterion(x_init, x_rec)
                loss_rec_all += loss_rec
            loss_rec = loss_rec_all
        elif self._remask_method == "fixed":
            rep = self.fixed_remask(g, origin_rep, mask_nodes)
            x_rec = self.decoder(pre_use_g, rep)[mask_nodes]
            x_init = x[mask_nodes]
            loss_rec = self.criterion(x_init, x_rec)
        else:
            raise NotImplementedError

        loss = loss_rec + self._lam * loss_latent

        if epoch >= self._delayed_ema_epoch:
            self.ema_update()
        return loss

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
            # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    def get_encoder(self):
        #self.encoder.reset_classifier(out_size)
        return self.encoder
    
    def reset_encoder(self, out_size):
        self.encoder.reset_classifier(out_size)
 
    def output_grad(self):
        grad_dict = {}
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_dict[n] = p.grad.abs().mean().item()
        return grad_dict
    
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # exclude isolated nodes
        # isolated_nodes = torch.where(g.in_degrees() <= 1)[0]
        # mask_nodes = perm[: num_mask_nodes]
        # mask_nodes = torch.index_fill(torch.full((num_nodes,), False, device=device), 0, mask_nodes, True)
        # mask_nodes[isolated_nodes] = False
        # keep_nodes = torch.where(~mask_nodes)[0]
        # mask_nodes = torch.where(mask_nodes)[0]
        # num_mask_nodes = mask_nodes.shape[0]

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)
    
    def random_remask(self,g,rep,remask_rate=0.5):
        
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes: ]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes

    def fixed_remask(self, g, rep, masked_nodes):
        rep[masked_nodes] = 0
        return rep
