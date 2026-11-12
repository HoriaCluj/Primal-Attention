"""Quick check that the training code works with the installed PyTorch/timm versions (CPU, about 1 min).

Exercises what main_combo.py uses, short of a full training run: model construction, train and eval
transforms on a real image, Mixup, timm's optimizer, scheduler and loss scaler, and one forward/backward
pass with the KSVD loss. Run it after building an environment, locally or on wICE:

    <env>/bin/python analysis/imagenet_eta_ablation/check_env.py [--data <imagenet-100 folder>] [--checkpoint <file>]
"""
import argparse
import glob
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.normpath(os.path.join(HERE, "..", "..", "PrimalAttention_CV"))
sys.path.insert(0, CV)

import torch  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=os.path.join(HERE, "..", "..", "data", "imagenet-100-smoke"))
parser.add_argument("--checkpoint", default="", help="optional checkpoint.pth to test loading")
cli = parser.parse_args()

warnings.simplefilter("ignore")
import timm  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from timm.data import Mixup  # noqa: E402
from timm.loss import SoftTargetCrossEntropy  # noqa: E402
from timm.optim import create_optimizer  # noqa: E402
from timm.scheduler import create_scheduler  # noqa: E402
from timm.utils import ModelEma, NativeScaler  # noqa: E402

import models_combo  # noqa: E402
from datasets import build_transform  # noqa: E402
from main_combo import get_args_parser  # noqa: E402

print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| timm", timm.__version__)
args = get_args_parser().parse_args([
    "--model", "primal_small_patch16_224", "--data-path", cli.data, "--data-set", "IMNET-100",
    "--low-rank", "96", "--rank-multi", "10", "--num-ksvd-layer", "1", "--eta", "0.05", "--batch-size", "4"])
args.nb_classes = 100

image_files = sorted(glob.glob(os.path.join(cli.data, "train", "*", "*.JPEG")))[:4]
assert image_files, "no images under {}/train".format(cli.data)
train_tf, eval_tf = build_transform(True, args), build_transform(False, args)
images = torch.stack([train_tf(Image.open(f).convert("RGB")) for f in image_files])
eval_image = eval_tf(Image.open(image_files[0]).convert("RGB"))
print("1. transforms ok: train batch", tuple(images.shape), "| eval image", tuple(eval_image.shape))

mixup = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=args.mixup_prob,
              switch_prob=args.mixup_switch_prob, mode=args.mixup_mode, label_smoothing=args.smoothing,
              num_classes=args.nb_classes)
if torch.cuda.is_available():   # timm's Mixup builds its targets on CUDA, as in training (engine_primal.py:34-35)
    images, targets = mixup(images.cuda(), torch.arange(4).cuda())
    images, targets = images.cpu(), targets.cpu()
    print("2. mixup ok: targets", tuple(targets.shape))
else:
    targets = torch.nn.functional.one_hot(torch.arange(4), 100).float()
    print("2. mixup skipped: needs a GPU (normal on a login node)")

model = models_combo.primal_small_patch16_224(low_rank=96, rank_multi=10, num_ksvd_layer=1, num_classes=100,
                                              drop_rate=0.0, drop_path_rate=args.drop_path)
ema = ModelEma(model, decay=args.model_ema_decay, device="cpu", resume="")
optimizer = create_optimizer(args, model)
scheduler, _ = create_scheduler(args, optimizer)
scaler = NativeScaler()
print("3. model, EMA, optimizer ({}), scheduler ({}) ok | params {:,}".format(
    type(optimizer).__name__, type(scheduler).__name__, sum(p.numel() for p in model.parameters())))

model.train()
outputs, score_list, lambda_list = model(images)
loss = SoftTargetCrossEntropy()(outputs, targets)
s, lam = score_list[0], torch.diag_embed(lambda_list[0])
loss_e = torch.mean((torch.einsum('...nd,...ds->...ns', s[0], lam.unsqueeze(0).repeat(s[0].size(0), 1, 1, 1))).norm(dim=-1, p=2) ** 2) / 2
loss_r = torch.mean((torch.einsum('...nd,...ds->...ns', s[1], lam.unsqueeze(0).repeat(s[1].size(0), 1, 1, 1))).norm(dim=-1, p=2) ** 2) / 2
trace = torch.einsum('...ps,...pd->...sd', s[2], s[3].type_as(s[2])).mean(dim=0).trace()
loss = loss + args.eta * (loss_e + loss_r - trace) ** 2
optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step(1)
ema.update(model)
print("4. forward/backward/step ok: loss {:.4f}, Lambda grad norm {:.2e}".format(
    loss.item(), model.blocks[-1].attn.Lambda.grad.norm().item()))

if cli.checkpoint:
    state = torch.load(cli.checkpoint, map_location="cpu")
    missing = model.load_state_dict(state["model"])
    print("5. checkpoint loads ok: epoch", state["epoch"], "|", missing)
print("all checks passed")
