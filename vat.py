import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, framework='dgl', criterion=None, xi=1e-4, eps=2.5, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.framework = framework
        self.criterion = criterion

    def forward(self, model, x):
        if self.framework == 'dgl':
            bg = x[0]
            nodefea = x[1]
            edgefea = x[2]

            with torch.no_grad():
                pred, _ = model(bg, nodefea, edgefea)

            # prepare random unit tensor
            dn = torch.rand(nodefea.shape).sub(0.5).to(nodefea.device)
            de = torch.rand(edgefea.shape).sub(0.5).to(edgefea.device)
            dn = _l2_normalize(dn)
            de = _l2_normalize(de)

            with _disable_tracking_bn_stats(model):
                # calc adversarial direction
                for _ in range(self.ip):
                    dn.requires_grad_()
                    de.requires_grad_()
                    pred_hat, _ = model(bg, nodefea + self.xi * dn, edgefea + self.xi * de)
                    adv_distance = self.criterion(pred_hat, pred)
                    adv_distance = adv_distance.mean()
                    # logp_hat = F.log_softmax(pred_hat, dim=1)
                    # adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                    adv_distance.backward()
                    dn = _l2_normalize(dn.grad)
                    de = _l2_normalize(de.grad)
                    model.zero_grad()

                # calc LDS
                rn_adv = dn * self.eps
                re_adv = de * self.eps
                pred_hat,_ = model(bg, nodefea + rn_adv, edgefea + re_adv)
                lds = self.criterion(pred_hat, pred)
        else:
            if self.framework == 'geometric':
                z = x[0]
                pos = x[1]
                batch = x[2]
                # model(z, pos, batch)#[z, pos, batch]
                with torch.no_grad():
                    pred, _ = model(z, pos, batch)

                # prepare random unit tensor
                # dn = torch.rand(nodefea.shape).sub(0.5).to(nodefea.device)
                dn = torch.rand(pos.shape).sub(0.5).to(pos.device)
                # dn = _l2_normalize(dn)
                # de = _l2_normalize(de)

                with _disable_tracking_bn_stats(model):
                    # calc adversarial direction
                    for _ in range(self.ip):
                        dn.requires_grad_()
                        # de.requires_grad_()
                        pred_hat, _ = model(z, pos + self.xi * dn, batch)
                        adv_distance = self.criterion(pred_hat, pred)
                        adv_distance = adv_distance.mean()
                        # logp_hat = F.log_softmax(pred_hat, dim=1)
                        # adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                        adv_distance.backward()
                        dn = _l2_normalize(dn.grad)
                        # de = _l2_normalize(de.grad)
                        model.zero_grad()

                    # calc LDS
                    rn_adv = dn * self.eps
                    pred_hat, _ = model(z, pos + rn_adv, batch)
                    lds = self.criterion(pred_hat, pred)
        return lds
