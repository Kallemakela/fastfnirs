"""
Implementation adapted from:

https://github.com/tufts-ml/fNIRS-mental-workload-classifiers/blob/b5199d6184e659152d1fe650db48eba53a221186/helpers/models.py

Other implementations:

https://github.com/YeZiyi1998/DL4EEG-Classification/blob/master/model/eegnet.py

https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb

"""

import numpy as np
import matplotlib.pyplot as plt
from fnemo.utils.plotting import plot_chs
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from collections import defaultdict


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )

        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(pl.LightningModule):
    def __init__(
        self,
        n_chs=8,
        num_timesteps=60,
        in_channels=1,
        kernel_size_1=3,
        stride_1=1,
        num_classes=2,
        F1=4,
        D=2,
        F2_mult=2,
        p_dropout=0.5,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.F2 = F1 * D * F2_mult

        # Temporal convolution
        self.firstConv = nn.Sequential(
            # PlotLayer("Input", type="time_multi"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=F1,
                kernel_size=(1, kernel_size_1),
                stride=(1, stride_1),
                padding=(0, stride_1),
                bias=False,
            ),
            #'same' padding: used by the author;
            # kernel_size=(1,3): "filter length chosen to be half the sampling rate" - author
            nn.BatchNorm2d(
                num_features=F1,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            # PlotLayer("TemporalConv", type="time_multi"),
        )

        self.depthwiseConv = nn.Sequential(
            Conv2dWithConstraint(
                F1, F1 * D, kernel_size=(n_chs, 1), stride=(1, 1), groups=F1, bias=False
            ),
            #'valid' padding: used by the author;
            # kernel_size = (n_chs, 1): used by the author
            # PrintShapeLayer('c2'),
            nn.BatchNorm2d(
                F1 * D, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ELU(),  # used by author
            # nn.AvgPool2d(kernel_size=(1, 4)), #kernel_size=(1,4) used by author
            nn.AvgPool2d(kernel_size=(1, 5 - stride_1)),
            # PlotLayer("DepthwiseConv", type="time_multi"),
            nn.Dropout(p=p_dropout),
            # PrintShapeLayer('p2')
        )

        # depthwise convolution follow by pointwise convolution (pointwise convolution is just Conv2d with 1x1 kernel)
        self.separableConv = nn.Sequential(
            nn.Conv2d(
                F1 * D,
                self.F2,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
                groups=F1 * D,
                bias=False,
            ),
            # PrintShapeLayer("c3"),
            # PlotLayer("SepConv", type="time_multi"),
            nn.Conv2d(self.F2, self.F2, kernel_size=1, bias=False),
            # PrintShapeLayer("c4"),
            # PlotLayer("SepConv2", type="time"),
            nn.BatchNorm2d(
                self.F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            #             nn.ReLU(),
            nn.ELU(),  # use by author
            nn.AvgPool2d(kernel_size=(1, 8)),  # kernel_size=(1,8): used by author
            nn.Dropout(p=p_dropout),
            # PrintShapeLayer("p3"),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.F2, out_features=num_classes, bias=True)
        )

        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.val_step_outputs = []
        self.val_metrics = defaultdict(lambda: defaultdict(float))

    def forward(self, x):
        batch_size = x.shape[0]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.reshape(batch_size, -1)
        x = self.classifier(x)
        return x

    def step(self, batch):
        # x, y, _ = batch
        x, y = batch
        y_hat = self.forward(x)
        return {"y_hat": y_hat}

    def training_step(self, batch, batch_idx):
        # x, y, _ = batch
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        # x, y, _ = batch
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        self.val_step_outputs.append(
            {"y_hat": y_hat, "y": y, "dataloader_idx": dataloader_idx}
        )
        return loss

    def predict_step(
        self, batch: torch.Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Any:
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def on_validation_epoch_end(self):
        logits, y = defaultdict(list), defaultdict(list)
        for output in self.val_step_outputs:
            dataloader_idx = output["dataloader_idx"]
            logits[dataloader_idx].append(output["y_hat"])
            y[dataloader_idx].append(output["y"])

        for dataloader_idx in logits.keys():
            logits[dataloader_idx] = torch.cat(logits[dataloader_idx])
            y[dataloader_idx] = torch.cat(y[dataloader_idx])
            y_pred = logits[dataloader_idx].argmax(dim=-1)
            # metric = f1_score(
            #     y[dataloader_idx].cpu().numpy(), y_pred.cpu().numpy(), average="macro"
            # )
            metric = (
                (y[dataloader_idx] == y_pred).float().mean().item()
            )
            save_epoch = (
                self.current_epoch + 1
                if not hasattr(self, "pre_val_done") or self.pre_val_done
                else -1
            )
            self.val_metrics[dataloader_idx][save_epoch] = metric

        self.val_step_outputs.clear()

    def pred_proba(self, x):
        return F.softmax(self(x), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        self.step_size = 20
        self.gamma = 0.7
        lrs = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": lrs}
