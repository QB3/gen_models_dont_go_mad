import os
import numpy as np
import shutil
from compute_metrics import compute_metrics
import wandb
import time

import torch
import subprocess
from utils_cfm import train_cfm, generate_cfm

OUT_PATH = "."  # TODO modify your output path


def train_ddpm(n_retrain, network_path, dataset_path, num_epochs, out_path):
    """
    Function to train/finetune a network with DDPM, using the ddpm-torch package.

    Parameters
    ----------
    n_retrain: int
        Number of retraining steps.

    network_path: string
        Path to the pretrained network.

    dataset_path: string
        Path to the dataset.

    num_epochs: int
        Number of epochs used for finetuning.

    out_path: string
        Path to the directory where to store the finetuned network.

    Returns
    -------
    network_path: string
        Path to the finetuned network.
    """

    print(f"Training on {dataset_path}")
    full_out_network_path = os.path.join(out_path, str(n_retrain), "models")
    os.makedirs(full_out_network_path, exist_ok=True)
    checkpoint_name = "train_checkpoint"

    args = [
        "--train-device=cuda:0",
        f"--root={dataset_path}",
        f"--chkpt-path={network_path}",
        f"--resume",
        f"--epochs={num_epochs}",
        f"--chkpt-dir={full_out_network_path}",
        f"--chkpt-name={checkpoint_name}",
        f"--chkpt-intv={num_epochs}"
    ]

    os.chdir("../ddpm-torch")
    p = subprocess.Popen(["python", "train.py"] + args)
    p.wait()
    p.kill()

    network_path = os.path.join(
        full_out_network_path, "cifar10", checkpoint_name)
    print(f"Finished training. Network path is {network_path}")
    return network_path


def generate_ddpm(n_retrain, network_path, output_path, num_gen):
    """
    Function to generate images with DDPM, using the ddpm-torch package.

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
    print(f"Generating samples from {network_path}")
    gen_path = os.path.join(output_path, str(n_retrain), "gen_samples")
    os.makedirs(gen_path, exist_ok=True)
    args = [
        f"--chkpt-path={network_path}",
        "--use-ddim",  # to change for DDPM
        "--skip-schedule=quadratic",
        "--subseq-size=100",
        "--suffix=_ddim",
        f"--total-size={num_gen}",
        # --num-gpus 4  # to add back for multigpu
        f"--save-dir={gen_path}"
    ]

    os.chdir("../ddpm-torch")

    p = subprocess.Popen(["python", "generate.py"] + args)
    p.wait()
    p.kill()

    print(f"Finished generating samples to {gen_path}")
    return gen_path


def train(
        n_retrain, network_path, dataset_path, num_mimg, out_path, dataset_name,
        nproc_per_node):
    """
    Function to train/finetune a network with EDM, using the edm package.

    Parameters
    ----------
    n_retrain: int
        Number of retraining steps.

    network_path: string
        Path to the pretrained network.

    dataset_path: string
        Path to the dataset.

    num_mimg: int
        Number of images to go see duting finetuning: 50k corresponds to one epoch for CIFAR, 70k corresponds to one epoch for FFHQ.

    out_path: string
        Path to the directory where to store the finetuned network.

    dataset_name: string
        Name of the dataset, cifar or ffhq.

    nproc_per_node: int
        Number of GPU to use to parallelize training.

    Returns
    -------
    network_path: string
        Path to the finetuned network.
    """
    print(f"Training on {dataset_path}")
    full_out_network_path = os.path.join(out_path, str(n_retrain), "models")
    os.makedirs(full_out_network_path, exist_ok=True)

    if dataset_name == "cifar":
        args = [
            f"--outdir={full_out_network_path}",
            "--batch=256",
            "--nosubdir",
            "--cond=0",  # no conditioning
            "--arch=ddpmpp",
            f"--data={dataset_path}",
            f"--transfer={network_path}",
            f"--duration={num_mimg}",
        ]
    elif dataset_name == "ffhq":
        args = [
            f"--outdir={full_out_network_path}",
            # "--batch=128",  # otherwise memory error
            "--batch=256",  # Be careful to memory error
            "--nosubdir",
            "--cond=0",  # no conditioning
            "--arch=ddpmpp",
            f"--data={dataset_path}",
            f"--transfer={network_path}",
            f"--duration={num_mimg}",
            "--cres=1,2,2,2",
            "--lr=2e-4",
            "--dropout=0.05",
            "--augment=0.15"
        ]
    else:
        raise NotImplementedError("%s not handled" % dataset_name)

    if nproc_per_node <= 1:
        p = subprocess.Popen(
            ["torchrun", "--standalone", "../edm/train.py"] + args)
    else:
        all_args=[
            "torchrun", "--standalone", "--nproc_per_node=%i"%nproc_per_node,
            "../edm/train.py"] + args
        p = subprocess.Popen(all_args)

    p.wait()
    p.kill()

    num_kimg = int(num_mimg * 1000)
    num_kimg_str = str(num_kimg)
    while len(num_kimg_str) < 6:
        num_kimg_str = "0" + num_kimg_str

    network_path = os.path.join(
        full_out_network_path, f"network-snapshot-{num_kimg_str}.pkl"
    )
    print(f"Finished training. Network path is {network_path}")
    return network_path


def generate(
        n_retrain, network_path, output_path, num_gen, dataset_name,
        nproc_per_node):
    """
    Function to generate images with EDM, using the edm package.

    Parameters
    ----------
    n_retrain: int
        Number of retraining steps.

    network_path: string
        Path to the pretrained network.

    output_path: string
        Path to the directory where to save the generated images.

    num_gen: int
        Number of images to generate.

    dataset_name: string
        Name of the dataset, cifar or ffhq.

    nproc_per_node: int
        Number of GPU to use to parallelize training.

    Returns
    -------
    gen_path: string
        Path to the generated images.
    """
    print(f"Generating samples from {network_path}")
    gen_path = os.path.join(output_path, str(n_retrain), "gen_samples")
    os.makedirs(gen_path, exist_ok=True)
    time.sleep(15)

    if dataset_name == "cifar":
        args = [
            f"--outdir={gen_path}",
            f"--seeds=0-{num_gen-1}",
            "--batch=128",
            f"--network={network_path}",
        ]
    elif dataset_name == "ffhq":
        args = [
            f"--outdir={gen_path}",
            f"--seeds=0-{num_gen-1}",
            "--batch=128",
            "--steps=40",  # specific to ffhq
            f"--network={network_path}",
        ]
    else:
        raise NotImplementedError("%s not handled" % dataset_name)

    if nproc_per_node > 1:
        p = subprocess.Popen([
            # "python",
            "torchrun",
            "--standalone",
            "--nproc_per_node=%s" % nproc_per_node,
            "../edm/generate.py"] + args)
    else:
        p = subprocess.Popen(["python", "../edm/generate.py"] + args)


    p.wait()
    p.kill()

    print(f"Finished generating samples to {gen_path}")
    return gen_path


def mix(
        n_retrain, orig_folder, gen_folder, output_path, prop_gen_data,
        model_name, fully_synthetic=False):
    """
    Function to mix the original training data and the synthetically generated data.

    Parameters
    ----------
    n_retrain: int
        Number of retraining steps.

    orig_folder: string
        Path to the original data.

    gen_folder: string
        Path to the generated data.

    output_path: string
        Path to the directory where to save the mixed data.

    prop_gen_data: float
        Proportion of generated data to use, usually between 0 and 1.

    model_name: string
        Name of the model, edm, ddpm, or otcfm.

    fully_synthetic: bool, optional (default=False)
        Weather or not to retrain only on synthetic data.

    Returns
    -------
    mixed_path: string
        Path to the mixed dataset.
    """
    print("Mixing samples")
    mixed_path = os.path.join(
        output_path, str(n_retrain), "mixed_samples", "mixed_samples")
    os.makedirs(mixed_path, exist_ok=True)

    orig_imgs = os.listdir(orig_folder)
    orig_imgs = [os.path.join(orig_folder, img) for img in orig_imgs]

    # load generated image paths
    gen_imgs = os.listdir(gen_folder)
    gen_imgs = [os.path.join(gen_folder, img) for img in gen_imgs]
    n_gen_samples = int(len(gen_imgs) * prop_gen_data)
    # subsample generated image paths
    if fully_synthetic:
        train_files = gen_imgs
    else:
        sub_gen_imgs = np.random.choice(gen_imgs, n_gen_samples, replace=False)
        train_files = np.concatenate([orig_imgs, sub_gen_imgs])

    for i, file in enumerate(train_files):
        new_file_path = os.path.join(mixed_path, f"{i}.png")
        shutil.copyfile(file, new_file_path)

    print(
        f"Mixed samples from {orig_folder}, {gen_folder} to create folder of {len(train_files)} at {mixed_path}"
    )
    if model_name in ('ddpm', 'otcfm'):
        return os.path.join(output_path, str(n_retrain), "mixed_samples")
    else:
        return mixed_path


def create_dataset(n_retrain, mixed_path, out_path):
    """
    Function to create the EDM-specific dataset (for efficiency).

    Parameters
    ----------
    n_retrain: int
        Number of retraining steps.

    mixed_path: string
        Path to the mixed dataset.

    out_path: string
        Path to the directory where to save the mixed data.

    Returns
    -------
    dataset_path: string
        Path to EDM-specific dataset.

    """
    print("Create dataset")
    dataset_path = os.path.join(out_path, str(n_retrain), "mixed_dataset.zip")
    args = [f"--source={mixed_path}", f"--dest={dataset_path}"]

    p = subprocess.Popen(["python", "../edm/dataset_tool.py"] + args)
    p.wait()
    p.kill()
    print(f"Created dataset from {mixed_path} at {dataset_path}")
    return dataset_path


def pregenerate_data(args):
    """
    Pregenerate data once for all with the pretrained model.
    TODO rm dependance to CIFAR / FFHQ.
    In other words, we fintune with multiple level of synhtetic data, however, the first pretrained model is the same for all the level of synthetic data.
    Hence one can pregenerate images with the first common pretrained network.
    """
    out_path = os.path.join(OUT_PATH, args.name)
    network_path = args.network
    gen_path = generate(
        0, args.network,
        "/network/scratch/q/quentin.bertrand/perfgen/experiments/ffhq_pretrain_pregen/pregen/",
        args.num_gen, args.dataset_name)
    # gen_path = generate(
    #     0, args.network,
    #     "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_pretrain_pregen/pregen/",
    #     args.num_gen)


def iter_retrain(args):
    out_path = os.path.join(OUT_PATH, args.name)
    for iter in range(args.n_retrain + 1):
        if iter == 0:
            network_path = args.network
            if args.pregen_dataset == "":
                # Generate samples
                gen_path = generate(0, network_path, out_path, args.num_gen)
            else:
                # Use pregenerated samples, useful for iter 0
                gen_path = args.pregen_dataset
        else:
            if args.model_name == "edm":
                network_path = train(
                    iter, network_path, dataset_path_edm, args.num_mimg, out_path, dataset_name=args.dataset_name,
                    nproc_per_node=args.nproc_per_node)
                gen_path = generate(
                    iter, network_path, out_path, args.num_gen, args.dataset_name,
                    nproc_per_node=args.nproc_per_node)
            elif args.model_name == "ddpm":
                n_samples = 50_000 * (1 + args.prop_gen_data)
                num_epochs = np.round(args.num_mimg * 10 ** 6 / n_samples)
                num_epochs = int(num_epochs)
                num_epochs = max(num_epochs, 1)
                network_path = train_ddpm(
                    iter, network_path, mixed_dataset_path, num_epochs, out_path)
                gen_path = generate_ddpm(
                    iter, network_path, out_path, args.num_gen)
            elif args.model_name == "otcfm":
                # TODO set num steps
                batchsize = 128
                num_steps = (args.num_mimg * 10**6) // batchsize
                num_steps = int(num_steps)
                network_path = train_cfm(
                    iter, network_path, mixed_dataset_path, num_steps, out_path)
                gen_path = generate_cfm(
                    iter, network_path, out_path, args.num_gen)

        if args.compute_metrics:
            files = os.listdir(gen_path)
            for i in range(10):
                path_to_img = os.path.join(gen_path, files[i])
                wandb.log({"example": wandb.Image(path_to_img)})
            metrics = compute_metrics(gen_path, dataset_name=args.dataset_name)
            torch.cuda.empty_cache()
            for keys in metrics.keys():
                wandb.log({"eval"+str(keys): metrics[keys]})

        if args.fully_synthetic:
            print("Using only self-generated data at each retraining")
            # mixed_dataset_path = gen_path

        mixed_dataset_path = mix(
            iter, args.train_dataset, gen_path, out_path,
            args.prop_gen_data, args.model_name,
            fully_synthetic=args.fully_synthetic)

        if args.model_name == "edm":
            dataset_path_edm = create_dataset(iter, mixed_dataset_path, out_path)


if __name__ == "__main__":
    network_path = "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_ddpm_pretain_pregen/cifar10_1000.pt"
    output_path = "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_ddpm_pretain_pregen/pregen/"
    num_gen = 50_000
    generate_ddpm(0, network_path, output_path, num_gen)
    compute_metrics(
            "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_ddpm_pretain_pregen/pregen/0/gen_samples/eval/cifar10/cifar10_1000_ddim/", 'cifar')
