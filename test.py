import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torchmetrics as tm

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
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
try: from models.ours_net import OursNetSeg
except ImportError: OursNetSeg = None

try: from models.ours_net_nosem import OursNetSegNoSem
except ImportError: OursNetSegNoSem = None
try: from models.ours_net_nodp import OursNetSegNoDP
except ImportError: OursNetSegNoDP = None
try: from models.ours_net_noattn import OursNetSegNoAttn
except ImportError: OursNetSegNoAttn = None
try: from models.ours_net_noall import OursNetSegNoAll
except ImportError: OursNetSegNoAll = None
try: from models.ours_net_k200_400 import OursNetSegK200_400
except ImportError: OursNetSegK200_400 = None
try: from models.ours_net_k400_800 import OursNetSegK400_800
except ImportError: OursNetSegK400_800 = None


# ========================================================================
# 模型工厂（干净版）
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
        raise ValueError(f"不支持的 Method ID: {method_id}。当前支持 1-14。")

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

        self.val_acc = tm.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_miou = tm.JaccardIndex(task="multiclass", num_classes=self.num_classes)
        self.test_dice = tm.F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        self.test_class_acc = tm.Accuracy(task="multiclass", num_classes=self.num_classes, average='none')

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch['pos'], batch['x'], batch['y']
        return batch[0], batch[1], batch[2]

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


# ========================================================================
# 数据集获取（jaw 过滤）
# ========================================================================
def get_dataset(train_test_split=1, jaw='both') -> Dataset:
    test = Teeth3DSDataset("data/3dteethseg", processed_folder='processed', verbose=True,
                           pre_transform=PreTransform(classes=17), post_transform=None,
                           in_memory=False, force_process=False, is_train=False,
                           train_test_split=train_test_split)

    if jaw not in ('upper', 'lower', 'both'):
        raise ValueError(f"jaw 必须是 upper / lower / both 之一，实际得到: {jaw}")

    if jaw in ('upper', 'lower'):
        n_before = len(test.file_names)
        test.file_names = [f for f in test.file_names if jaw in f]
        print(f"[Jaw Filter] jaw='{jaw}': {n_before} -> {len(test.file_names)} samples")

        if len(test.file_names) == 0:
            txt = f'data/3dteethseg/raw/testing_{jaw}.txt'
            print(f"[Jaw Filter] .pt 文件名中未含 '{jaw}'，尝试从 {txt} 重建列表...")
            with open(txt) as f:
                wanted = {f'data_{l.rstrip()}.pt' for l in f if l.strip()}
            test.file_names = [f for f in test.processed_file_names if f in wanted]
            print(f"[Jaw Filter] fallback 命中 {len(test.file_names)} samples")

        if len(test.file_names) == 0:
            raise RuntimeError(f"jaw='{jaw}' 过滤后样本为 0，请检查数据命名或 txt 文件。")
    else:
        print(f"[Jaw Filter] jaw='both'，不做过滤，共 {len(test.file_names)} samples")

    return test


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

    parser = argparse.ArgumentParser(description='Run testing')
    parser.add_argument('--method', type=int, choices=list(range(1, 15)), required=True,
                        help='测试方法编号 (与训练 ckpt 时的 method 一致)')
    parser.add_argument('--jaw', type=str, choices=['upper', 'lower', 'both'], default='both')
    parser.add_argument('--tb_save_dir', type=str, default='tensorboard_logs')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--experiment_version', type=str, default=None)
    parser.add_argument('--devices', nargs='+', required=True)
    parser.add_argument('--n_bit_precision', type=str, default="16")
    parser.add_argument('--train_test_split', type=int, default=1)
    parser.add_argument('--ckpt', type=str, required=True)

    args = parser.parse_args()
    print(f'=== 🎯 准备启动测试 | Method: {args.method} | Jaw: {args.jaw} ===')
    print(f'参数: {args}')

    test_dataset = get_dataset(args.train_test_split, jaw=args.jaw)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                 shuffle=False, num_workers=0)

    pl_model = LitToothSegmentationWrapper.load_from_checkpoint(
        args.ckpt, 
        method_id=args.method
    )

    exp_name = args.experiment_name if args.experiment_name else f'Method_{args.method}_test'
    version = args.experiment_version if args.experiment_version else args.jaw
    logger = CSVLogger(args.tb_save_dir, name=exp_name, version=version)

    precision_val = "16-mixed" if args.n_bit_precision == "16" else (
        int(args.n_bit_precision) if args.n_bit_precision.isdigit() else args.n_bit_precision)

    trainer = pl.Trainer(
        accelerator='cuda',
        devices=[int(d) for d in args.devices],
        enable_progress_bar=True,
        logger=logger,
        precision=precision_val,
        deterministic=False
    )

    print("\n" + "=" * 50)
    print(f"📥 成功加载模型权重: {args.ckpt}")
    print(f"📂 测试颌型: {args.jaw}")
    print("=" * 50)

    trainer.test(model=pl_model, dataloaders=val_dataloader, verbose=True)