import argparse
import random
import numpy as np
import os

import torch
import torch.nn as nn
import torchmetrics as tm

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data.dataset import Dataset

from dataset.mesh_dataset import Teeth3DSDataset
from dataset.preprocessing import *

# ========================================================================
# 动态导入
# ========================================================================
try:
    from models.dilated_tooth_seg_network import DilatedToothSegmentationNetwork
except ImportError:
    try:
        from models.dilated_tooth_seg_network0 import DilatedToothSegmentationNetwork
    except ImportError:
        DilatedToothSegmentationNetwork = None

try: from models.pointnet import PointNetSeg
except ImportError: PointNetSeg = None
try: from models.pointnet2 import PointNet2Seg
except ImportError: PointNet2Seg = None
try: from models.tsgcnet import TSGCNetSeg
except ImportError: TSGCNetSeg = None
try: from models.meshsegnet import MeshSegNetSeg
except ImportError: MeshSegNetSeg = None
try: from models.pct import PCT
except ImportError: PCT = None
try: from models.dgcnn import DGCNNSeg
except ImportError: DGCNNSeg = None

# 原始完整模型 (Method 7)
try: from models.ours_net import OursNetSeg
except ImportError: OursNetSeg = None

# 消融实验模型 (Method 9 - 12)
try: from models.ours_net_nosem import OursNetSegNoSem
except ImportError: OursNetSegNoSem = None
try: from models.ours_net_nodp import OursNetSegNoDP
except ImportError: OursNetSegNoDP = None
try: from models.ours_net_noattn import OursNetSegNoAttn
except ImportError: OursNetSegNoAttn = None
try: from models.ours_net_noall import OursNetSegNoAll
except ImportError: OursNetSegNoAll = None

# 参数敏感性模型 (Method 13 - 14)
try: from models.ours_net_k200_400 import OursNetSegK200_400
except ImportError: OursNetSegK200_400 = None
try: from models.ours_net_k400_800 import OursNetSegK400_800
except ImportError: OursNetSegK400_800 = None


# ========================================================================
# 模型工厂（所有方法只传 num_classes / feature_dim）
# ========================================================================
def get_network(method_id: int, num_classes=17, feature_dim=24):
    networks = {
        1: DilatedToothSegmentationNetwork,
        2: PointNetSeg,
        3: PointNet2Seg,
        4: TSGCNetSeg,
        5: MeshSegNetSeg,
        6: PCT,
        7: OursNetSeg,
        8: DGCNNSeg,
        9: OursNetSegNoSem,
        10: OursNetSegNoDP,
        11: OursNetSegNoAttn,
        12: OursNetSegNoAll,
        13: OursNetSegK200_400,
        14: OursNetSegK400_800,
    }

    if method_id not in networks:
        raise ValueError(f"不支持 Method ID: {method_id} (支持 1-14)")

    NetClass = networks[method_id]
    if NetClass is None:
        raise NotImplementedError(f"未找到 Method {method_id} 对应的代码。")

    return NetClass(num_classes=num_classes, feature_dim=feature_dim)


# ========================================================================
# Lightning Wrapper
# ========================================================================
class LitToothSegmentationWrapper(pl.LightningModule):
    def __init__(self, method_id=1, num_classes=17, feature_dim=24, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.model = get_network(method_id, num_classes, feature_dim)
        self.ce_loss = nn.CrossEntropyLoss()

        self.train_acc = tm.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_miou = tm.JaccardIndex(task="multiclass", num_classes=self.num_classes)
        self.val_miou = tm.JaccardIndex(task="multiclass", num_classes=self.num_classes)
        self.test_dice = tm.F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        self.test_class_acc = tm.Accuracy(task="multiclass", num_classes=self.num_classes, average='none')

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch['pos'], batch['x'], batch['y']
        return batch[0], batch[1], batch[2]

    def training_step(self, batch, batch_idx):
        pos, x, y = self._unpack_batch(batch)
        x, y = x.float(), y.long()
        pred = self.model(x, pos).transpose(2, 1)
        loss = self.ce_loss(pred, y)
        self.train_acc(pred, y)
        self.train_miou(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_miou", self.train_miou, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pos, x, y = self._unpack_batch(batch)
        x, y = x.float(), y.long()
        pred = self.model(x, pos).transpose(2, 1)
        loss = self.ce_loss(pred, y)
        self.val_acc(pred, y)
        self.val_miou(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        pos, x, y = self._unpack_batch(batch)
        x, y = x.float(), y.long()
        pred = self.model(x, pos).transpose(2, 1)
        self.val_acc(pred, y)
        self.val_miou(pred, y)
        self.test_dice(pred, y)
        self.test_class_acc(pred, y)
        self.log("test_dice", self.test_dice, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        macro_dice = self.test_dice.compute()
        class_acc = self.test_class_acc.compute()
        class_names = [
            "Gum", "Central Incisor (UL)", "Lateral Incisor (UL)", "Canine (UL)", "1st Premolar (UL)", "2nd Premolar (UL)", "1st Molar (UL)", "2nd Molar (UL)", "3rd Molar (UL)",
            "Central Incisor (UR)", "Lateral Incisor (UR)", "Canine (UR)", "1st Premolar (UR)", "2nd Premolar (UR)", "1st Molar (UR)", "2nd Molar (UR)", "3rd Molar (UR)"
        ]
        print("\n" + "=" * 50)
        print(f"⭐ FINAL TEST RESULTS | Macro Dice (F1): {macro_dice.item() * 100:.2f}%")
        print("=" * 50)
        for i, acc in enumerate(class_acc):
            name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"{name:<25}: {acc.item() * 100:.2f}%")
        print("=" * 50 + "\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"}}


def get_dataset(train_test_split=1):
    train = Teeth3DSDataset("data/3dteethseg", processed_folder='processed', verbose=True,
                            pre_transform=PreTransform(classes=17), post_transform=None,
                            in_memory=False, force_process=False, is_train=True,
                            train_test_split=train_test_split)
    test = Teeth3DSDataset("data/3dteethseg", processed_folder='processed', verbose=True,
                           pre_transform=PreTransform(classes=17), post_transform=None,
                           in_memory=False, force_process=False, is_train=False,
                           train_test_split=train_test_split)
    return train, test


# ========================================================================
# 主入口
# ========================================================================
if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    torch.set_float32_matmul_precision('medium')
    random.seed(SEED)
    seed_everything(SEED, workers=True)

    parser = argparse.ArgumentParser(description='Run Training with Multiple Methods')
    parser.add_argument('--method', type=int, choices=list(range(1, 15)), default=7)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--tb_save_dir', type=str, default='tensorboard_logs')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--experiment_version', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--devices', nargs='+', required=True, default=[0])
    parser.add_argument('--n_bit_precision', type=str, default="16")
    parser.add_argument('--train_test_split', type=int, default=1)
    parser.add_argument('--ckpt', type=str, required=False, default=None)

    args = parser.parse_args()

    precision_val = "16-mixed" if args.n_bit_precision == "16" else (
        int(args.n_bit_precision) if args.n_bit_precision.isdigit() else args.n_bit_precision)

    tag = {
        7:  "[Method 7 ] 完整模型 (K1=300, K2=600)",
        9:  "[Method 9 ] 去掉语义构图 (w/o SAHM-Graph)",
        10: "[Method 10] 去掉双频空洞图卷积 (w/o Dual-Freq Dilated GCN)",
        11: "[Method 11] 去掉双域注意力 (w/o Dual-Domain Attention)",
        12: "[Method 12] 去掉all (w/o All)",
        13: "[Method 13] K1=200, K2=400 (参数灵敏性)",
        14: "[Method 14] K1=400, K2=800 (参数灵敏性)",
    }
    print(f'=== 🚀 准备启动实验 | Method: {args.method} | {tag.get(args.method, "")} ===')

    train_dataset, test_dataset = get_dataset(args.train_test_split)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True, drop_last=True, num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    pl_model = LitToothSegmentationWrapper(
        method_id=args.method, num_classes=17, feature_dim=24
    )

    exp_name = args.experiment_name if args.experiment_name else f'Method_{args.method}'
    logger = TensorBoardLogger(args.tb_save_dir, name=exp_name, version=args.experiment_version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        save_top_k=6,
        monitor="val_acc",
        mode='max',
        filename="best-epoch{epoch:02d}-val_acc{val_acc:.4f}"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='cuda',
        devices=[int(d) for d in args.devices],
        enable_progress_bar=True,
        logger=logger,
        precision=precision_val,
        callbacks=[checkpoint_callback],
        deterministic=False
    )

    trainer.fit(
        model=pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.ckpt
    )

    print("\n" + "=" * 50)
    print("🎯 训练阶段完成，正在测试...")
    trainer.test(model=pl_model, dataloaders=val_dataloader, ckpt_path="best")