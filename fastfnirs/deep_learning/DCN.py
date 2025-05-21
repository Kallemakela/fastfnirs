import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from collections import defaultdict

# https://github.com/tufts-ml/fNIRS-mental-workload-classifiers/blob/b5199d6184e659152d1fe650db48eba53a221186/helpers/models.py


class PrintShapeLayer(nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def forward(self, x):
        # print(self.name, x.shape)
        return x


class DeepConvNet(pl.LightningModule):
    def __init__(
        self,
        n_chs=8,
        num_timesteps=60,
        in_channels=1,
        num_classes=2,
        kernel_size=2,
        pool_size=2,
        pool_stride=2,
        stride=1,
        p_dropout=0.5,
        D1=25,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        padding = (kernel_size - 1) // 2

        b1_size = (num_timesteps + 2 * padding - kernel_size) // stride + 1
        b1_size = (b1_size - pool_size) // pool_stride + 1

        b2_size = (b1_size + 2 * padding - kernel_size) // stride + 1
        b2_size = (b2_size - pool_size) // pool_stride + 1

        b3_size = (b2_size + 2 * padding - kernel_size) // stride + 1
        b3_size = (b3_size - pool_size) // pool_stride + 1

        b4_size = (b3_size + 2 * padding - kernel_size) // stride + 1
        b4_size = (b4_size - pool_size) // pool_stride + 1

        # print(f"b1_size: {b1_size}")
        # print(f"b2_size: {b2_size}")
        # print(f"b3_size: {b3_size}")
        # print(f"b4_size: {b4_size}")

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=D1,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                bias=True,
            ),
            # PrintShapeLayer('block1_conv1'),
            nn.Conv2d(
                in_channels=D1,
                out_channels=D1,
                kernel_size=(n_chs, 1),
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=D1,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
            # PrintShapeLayer('block1_maxpool'),
        )

        self.block2 = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Conv2d(
                in_channels=D1,
                out_channels=D1 * 2,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=D1 * 2,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(),  # use by author
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
            # PrintShapeLayer('block2_maxpool'),
        )

        self.block3 = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Conv2d(
                in_channels=D1 * 2,
                out_channels=D1 * 4,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=D1 * 4,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(),  # use by author
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
            # PrintShapeLayer('block3_maxpool'),
        )

        self.block4 = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Conv2d(
                in_channels=D1 * 4,
                out_channels=D1 * 8,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=D1 * 8,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(),  # use by author
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
            # PrintShapeLayer('block4_maxpool'),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=D1 * 8,
                out_channels=num_classes,
                kernel_size=(1, b4_size),
                bias=True,
            ),
            # PrintShapeLayer('classifier_out'),
        )

        self.val_step_outputs = []
        self.val_metrics = defaultdict(lambda: defaultdict(float))

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        x = x.flatten(1)
        return x

    def step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        return {"y_hat": y_hat}

    def pred_proba(self, x):
        return F.softmax(self(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.step(batch)
        y_hat = output["y_hat"]
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def predict_step(
        self, batch: torch.Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Any:
        return self.step(batch)["y_hat"]

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        x, y = batch
        output = self.step(batch)
        y_hat = output["y_hat"]
        assert y_hat.shape[1] == self.hparams.num_classes
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        self.val_step_outputs.append(
            {"y_hat": y_hat, "y": y, "dataloader_idx": dataloader_idx}
        )
        return {"val_loss": loss}

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
            metric = f1_score(
                y[dataloader_idx].cpu().numpy(), y_pred.cpu().numpy(), average="macro"
            )
            save_epoch = (
                self.current_epoch + 1
                if not hasattr(self, "pre_val_done") or self.pre_val_done
                else -1
            )
            self.val_metrics[dataloader_idx][save_epoch] = metric

        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        self.step_size = 20
        self.gamma = 0.7
        lrs = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": lrs}
