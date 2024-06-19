"""
This code is mostly taken from the optimal transport flow matching toolbox
https://github.com/atong01/conditional-flow-matching/tree/main/examples/images/cifar10
"""

import copy
import os

from torchdiffeq import odeint

import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
from torchvision.utils import save_image
from tqdm import tqdm



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train_cfm(n_retrain, network_path, dataset_path, num_steps, out_path):
    """
    Function to train/finetune a network with OTCFM, using the conditional-flow-mathcing package.

    Parameters
    ----------
    n_retrain: int
        Number of retraining steps.

    network_path: string
        Path to the pretrained network.

    dataset_path: string
        Path to the dataset.

    num_steps: int
        Number of steps used for finetuning.

    out_path: string
        Path to the directory where to store the finetuned network.

    Returns
    -------
    model_path: string
        Path to the finetuned network.
    """

    full_out_network_path = os.path.join(out_path, str(n_retrain), "models")
    os.makedirs(full_out_network_path, exist_ok=True)

    # DATASETS/DATALOADER
    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    net_model, ema_model, optim = load_checkpoint(network_path)

    # Show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    # import ipdb; ipdb.set_trace()
    with trange(num_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), 1.0)  # new
            optim.step()
            # Maybe to change with only 20 epochs
            ema(net_model, ema_model, 0.9999)  # new

    model_path = os.path.join(full_out_network_path, f"model.pt")
    torch.save(
        {
            "net_model": net_model.state_dict(),
            "ema_model": ema_model.state_dict(),
            # "sched": sched.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
        },
        model_path
    )
    return model_path

def generate_cfm(
    n_retrain, network_path, output_path, num_gen):
    """
    Function to generate images with OTCFM, using the conditional-flow-mathcing package.

    Parameters
    ----------
    n_retrain: int
        Number of retraining steps.

    network_path: string
        Path to the pretrained network.

    out_path: string
        Path to the directory where to store the finetuned network.

    num_gen: int
        Number of images to generate (usually 50k for CIFAR and 70k for FFHQ).

    Returns
    -------
    gen_path: string
        Path to the generated images.
    """
    BATCH_SIZE = 512
    print(f"Generating samples from {network_path}")
    gen_path = os.path.join(output_path, str(n_retrain), "gen_samples/0/") # 0 is dummy folder for ImageFolder
    os.makedirs(gen_path, exist_ok=True)

    _, ema_model, _  = load_checkpoint(network_path)
    # ema_node = NeuralODE(ema_model, solver="euler", sensitivity="adjoint")

    ema_model.eval()
    for batch in tqdm(range(num_gen//BATCH_SIZE + 1)):
        with torch.no_grad():
            # traj = ema_node.trajectory(
            #     torch.randn(BATCH_SIZE, 3, 32, 32).to(device),
            #     t_span=torch.linspace(0, 1, 100).to(device),
            # )
            x = torch.randn(BATCH_SIZE, 3, 32, 32).to(device)
            t_span = torch.linspace(0, 1, 2).to(device)
            traj = odeint(
                ema_model, x, t_span, rtol=1e-5, atol=1e-5, method="dopri5"
            )
            traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
            traj = traj / 2 + 0.5

        for i in range(BATCH_SIZE):
            idx = batch * BATCH_SIZE + i
            if idx < num_gen:
                img_path = os.path.join(gen_path, f"{idx}.png")
                save_image(traj[i], img_path)

    return gen_path


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x



def create_objects():
    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=2e-4)

    return net_model, ema_model, optim

def load_checkpoint(checkpoint_path):
    net_model, ema_model, optim = create_objects()

    chkpt = torch.load(checkpoint_path)
    net_model.load_state_dict(chkpt["net_model"])
    ema_model.load_state_dict(chkpt["ema_model"])
    optim.load_state_dict(chkpt["optim"])

    # if FLAGS.parallel:
    #     net_model = torch.nn.DataParallel(net_model)
    #     ema_model = torch.nn.DataParallel(ema_model)
    return net_model, ema_model, optim



# generate(0, "/home/mila/m/marco.jiralerspong/scratch/recursive_edm/conditional-flow-matching/examples/cifar10/otcfm_cifar10_weights_step_400000.pt", "base_gen_samples", 50000)
