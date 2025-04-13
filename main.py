import os
import time
import warnings

import numpy as np
import torch
from dotenv import load_dotenv
from pymongo import MongoClient
from rich.pretty import pprint
from sklearn.metrics import davies_bouldin_score, silhouette_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from src.datasets import Berka
from src.db import get_config_by_id, get_results_missing_run, log_run_result
from src.drivers import (
    calculate_clusters_from_latents,
    calculate_clusters_from_layer,
    pretrain_autoencoder,
    train_clustering_layer,
)
from src.factories import (
    decoder_factory,
    embedding_factory,
    encoder_factory,
    metric_factory,
    pretext_factory,
)
from src.models import Autoencoder

load_dotenv()
warnings.filterwarnings("ignore")


def process(
    ae_name,
    emb_name,
    pretext_loss,
    cluster_loss,
    metric,
    embedding_dim,
    n_clusters,
    dim_reduction,
    reduced_dim,
    seq_len,
    train_loader,
    test_loader,
    device,
):
    encoder = encoder_factory(
        ae_name=ae_name,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
    )
    embedding = embedding_factory(
        ae_name=emb_name,
        pretext_loss=pretext_loss,
        encoder_output_dim=encoder.output_dim,
        embedding_dim=embedding_dim,
    )
    decoder = decoder_factory(
        ae_name=ae_name,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
    )
    pretext_loss_fn = pretext_factory(pretext=pretext_loss)
    model = Autoencoder(
        encoder=encoder,
        embedding=embedding,
        decoder=decoder,
        pretext_loss_fn=pretext_loss_fn,
        device=device,
    )

    start_time = time.time()
    pretrain_autoencoder(model, train_loader, device=device)

    if cluster_loss is not None:
        metric_np, metric_torch = metric_factory(metric=metric)
        model = train_clustering_layer(
            encoder=model.encoder,
            embedding=model.embedding,
            decoder=model.decoder,
            metric_np=metric_np,
            metric_torch=metric_torch,
            n_clusters=n_clusters,
            loader=train_loader,
            device=device,
        )
        training_time = time.time() - start_time

        with torch.no_grad():
            torch.cuda.empty_cache()
            train_clusters = calculate_clusters_from_layer(model, train_loader, device)
            torch.cuda.empty_cache()
            test_clusters = calculate_clusters_from_layer(model, test_loader, device)
    else:
        with torch.no_grad():
            training_time = time.time() - start_time
            torch.cuda.empty_cache()
            train_clusters = calculate_clusters_from_latents(
                model=model,
                loader=train_loader,
                k=n_clusters,
                dim_reduction=dim_reduction,
                lower_dim=reduced_dim,
                device=device,
            )
            torch.cuda.empty_cache()
            test_clusters = calculate_clusters_from_latents(
                model=model,
                loader=train_loader,
                k=n_clusters,
                dim_reduction=dim_reduction,
                lower_dim=reduced_dim,
                device=device,
            )
    return train_clusters, test_clusters, training_time


def load_dataset(loader):
    xs = []
    for x, _ in loader:
        xs.append(x)
    X = torch.concatenate(xs).squeeze(1).numpy()
    return X


if __name__ == "__main__":
    client = MongoClient(
        username=os.getenv("MONGO_USERNAME"),
        password=os.getenv("MONGO_PASSWORD"),
        authSource="admin",
    )
    db = client.result_db
    config_collection = db.config_collection
    result_collection = db.result_collection

    test_frac = 0.5
    batch_size = 32
    np.random.seed(0)
    dataset = Berka(
        account_path="data/Berka/account.csv", transaction_path="data/Berka/trans.csv"
    )

    for run_count in range(1, 6):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_frac * dataset_size))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train = load_dataset(train_loader)
        X_test = load_dataset(test_loader)

        run_name = f"run_{run_count}"

        results = get_results_missing_run(result_collection, run_name)
        doc_count = result_collection.count_documents({run_name: {"$exists": False}})

        results = [r for r in results]
        configs = [
            get_config_by_id(config_collection, res["config_id"]) for res in results
        ]

        for cfg, res in tqdm(zip(configs, results), total=doc_count):
            print("--------------------------------")
            print("Evaluating config:")
            print("--------------------------------")
            pprint(cfg)
            print("--------------------------------")

            try:
                clusters_train, clusters_test, train_time = process(
                    ae_name=cfg["autoencoder"],
                    emb_name=cfg["embedding"],
                    pretext_loss=cfg["pretext_loss"],
                    cluster_loss=cfg["cluster_loss"],
                    metric=cfg["metric"],
                    embedding_dim=cfg["embedding_dim"],
                    n_clusters=cfg["n_clusters"],
                    dim_reduction=cfg["dim_reduction"],
                    reduced_dim=cfg["reduced_dim"],
                    seq_len=675,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                )

                if len(np.unique(clusters_train)) == cfg["n_clusters"]:
                    SC_train = float(silhouette_score(X_train, clusters_train))
                    DBI_train = float(davies_bouldin_score(X_train, clusters_train))
                else:
                    SC_train = -1
                    DBI_train = -1

                if len(np.unique(clusters_test)) == cfg["n_clusters"]:
                    SC_test = float(silhouette_score(X_test, clusters_test))
                    DBI_test = float(davies_bouldin_score(X_test, clusters_test))
                else:
                    SC_test = -1
                    DBI_test = -1

            except Exception as e:
                SC_test = -2
                DBI_test = -2
                SC_train = -2
                DBI_train = -2
                train_time = 0
                print(f"Error: {str(e)}")

            try:
                log_run_result(
                    collection=result_collection,
                    run_name=run_name,
                    result_id=res["_id"],
                    sc_train=SC_train,
                    dbi_train=DBI_train,
                    sc_test=SC_test,
                    dbi_test=DBI_test,
                    train_time=train_time,
                )
                break
            except Exception as e:
                print(str(e))
                print("Could not log result to database")
                time.sleep(5)

            torch.cuda.empty_cache()
