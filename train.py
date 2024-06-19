import argparse
import wandb
from iterative_retraining import perfgen, pregenerate_data, debug_perfgen


dict_config = {}
"""
dict_config is a dict with keys (dataset, algorithm)
The values are (path to pretrained network, path to training data, path to pregenerated data)
"""
dict_config["cifar", "otcfm"] = (
    "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_cfm_pretrain_pregen/otcfm_cifar10_weights_step_400000.pt",
    "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_cfm_pretrain_pregen/cifar_png/cifar/train/",
    "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_cfm_pretrain_pregen/pregen/0/gen_samples/0/")
dict_config["cifar", "edm"] = (
    "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_pretrain_pregen/edm-cifar10-32x32-uncond-vp.pkl",
    "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_pretrain_pregen/cifar_png/cifar/train/",
    "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_pretrain_pregen/pregen")
dict_config["cifar", "ddpm"] = (
    "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_ddpm_pretain_pregen/cifar10_1000.pt",
    "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_ddpm_pretain_pregen/train/train",
    "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_ddpm_pretain_pregen/pregen/0/gen_samples/eval/cifar10/cifar10_1000_ddim/") #
dict_config["ffhq", "edm"] = (
        "/network/scratch/q/quentin.bertrand/perfgen/experiments/ffhq_pretrain_pregen/edm-ffhq-64x64-uncond-vp.pkl",
        "/network/scratch/q/quentin.bertrand/perfgen/experiments/ffhq_pretrain_pregen/ffhq_png/datasets/train/png/",
        "/network/scratch/q/quentin.bertrand/perfgen/experiments/ffhq_pretrain_pregen/pregen/0/gen_samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, help="Name of experiment", required=True)
    parser.add_argument("--prop_gen_data", type=float, required=True)
    parser.add_argument(
        "--dataset_name", type=str, help="Name of the dataset", required=True
    )
    parser.add_argument(
        "--model_name", type=str, help="Name of the model", default="edm"
    )  # ddpm or edm
    parser.add_argument(
        "--n_retrain", type=int, help="Number of recursive iterations", default=5
    )
    parser.add_argument(
        "--compute_metrics", action="store_true", help="Compute metrics or not")

    parser.add_argument(
        "--fully_synthetic", action="store_true", help="Use only synthetic data")

    parser.add_argument(
        "--num_mimg", type=float, help="Num mimg to train for", default=0.05
    )
    parser.add_argument("--num_gen", type=int, help="Num gen", default=50000)
    parser.add_argument("--nproc_per_node", type=int, help="Num gen", default=1)

    args = parser.parse_args()
    if args.fully_synthetic:
        print("Using only self-generated data at each retraining")
        args.name = "%s_%s_fully_synthetic_%.2f_%i" % (
            args.name, args.model_name, args.num_mimg, args.n_retrain)
    else:
        args.name = "%s_%s_%.3f_%.2f_%i" % (
            args.name, args.model_name, args.prop_gen_data, args.num_mimg,
            args.n_retrain)

    args.network, args.train_dataset, args.pregen_dataset = dict_config[
        args.dataset_name, args.model_name]

    run = wandb.init(
        project="Generative don't go mad",
        config={
            "n_retrain": args.n_retrain,
            "prop_gen_data": args.prop_gen_data,
            "num_gen": args.num_gen,
            "model": args.network,
        },
        name=args.name
    )
    # pregenerate_data(args)  # Used to pregenerate data
    perfgen(args)
