from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor

from fld.metrics.FID import FID
from fld.metrics.PrecisionRecall import PrecisionRecall

import torchvision

SAVE_PATH = "/network/scratch/q/quentin.bertrand/perfgen/experiments/data/features"


def compute_metrics(gen_path, dataset_name):
    """
    Function to compute the FID (Frechet Inception Distance), Precision, and Recall between the generated and synthetic dataset.

    Parameters
    ----------
    gen_path: string
        Path to the generated data.

    dataset_name: string
        Name of the dataset, cifar or ffhq


    Returns
    -------
    res: dict of float
        Dictionary with key FID, Precision, and Recall.
    """
    # Save path determines where features are cached (useful for train/test sets)
    print('Computing metrics')
    feature_extractor = InceptionFeatureExtractor(
        recompute=True,  # CIFAR training data features could be cached
        save_path=SAVE_PATH
        # save_path=False,  # To be double checked
        # save_path="data/features",
    )

    # FLS needs 3 sets of samples: train, test and gen
    if dataset_name == 'cifar':
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        train_dataset.name = "CIFAR10_train"  # Dataset needs a name to cache features
        ref_size = 50_000
    elif dataset_name == 'ffhq':
        # TODO adapt since FLD API has changed
        # train_dataset = SamplesDataset("ffhq_train", "/network/scratch/q/quentin.bertrand/perfgen/experiments/ffhq_pretrain_pregen/ffhq_png/datasets/train/png/")
        ref_size = 70_000
        train_dataset.name = ""

    train_feat = feature_extractor.get_features_from_dataset(train_dataset)
    gen_feat = feature_extractor.get_dir_features(gen_path, extension="png")

    metrics = [
        FID(mode="train", ref_size=ref_size),
        PrecisionRecall("Precision"),
        PrecisionRecall("Recall"),
    ]

    res = {}
    for metric in metrics:
        res[metric.name] = metric.compute_metric(
            train_feat, train_feat, gen_feat)

    print(res)
    return res


if __name__ == "__main__":
    res = compute_metrics(
        "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_cfm_pretrain_pregen/pregen/0/gen_samples", 'cifar')
    # res = compute_metrics(
    #     "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_ddpm_pretain_pregen/pregen/0/gen_samples/eval/cifar10/cifar10_1000_ddim/", 'cifar')
    print(res)
