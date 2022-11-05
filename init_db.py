from itertools import product

import pymongo
from bson.objectid import ObjectId
from pymongo import MongoClient

client = MongoClient("")


search_config = {
    "autoencoder": ["fcnn", "resnet", "lstm", "dtc"],
    "pretext_losses": ["mse", "multi_rec", "vae"],
    "cluster_losses": [None, "dtc"],
    "metrics": ["eucl", "cid"],
    "embedding_dim": [10],
    "n_clusters": [2, 4, 6],
    "dim_reduction": [None, "PCA", "UMAP"],
    "reduced_dim": [5],
}


def generate_configs():
    configs = []
    for cfg in product(*search_config.values()):
        cluster_loss = cfg[2]

        if cluster_loss is None:
            dim_reduction = None
            reduced_dim = None
            metric = None
        else:
            dim_reduction = cfg[6]
            reduced_dim = cfg[7]
            metric = cfg[3]

        configs.append(
            {
                "autoencoder": cfg[0],
                "embedding": cfg[1],
                "pretext_loss": cfg[1],
                "cluster_loss": cfg[2],
                "metric": metric,
                "embedding_dim": cfg[4],
                "n_clusters": cfg[5],
                "dim_reduction": dim_reduction,
                "reduced_dim": reduced_dim,
            }
        )
    return configs


if __name__ == "__main__":
    # Creating database
    db = client.result_db

    # Creating collections
    config_collection = db.config_collection
    result_collection = db.result_collection

    # Creating collection indexes
    config_collection.create_index(
        keys=[
            ("autoencoder", pymongo.ASCENDING),
            ("embedding", pymongo.ASCENDING),
            ("pretext_loss", pymongo.ASCENDING),
            ("cluster_loss", pymongo.ASCENDING),
            ("metric", pymongo.ASCENDING),
            ("embedding_dim", pymongo.ASCENDING),
            ("n_clusters", pymongo.ASCENDING),
            ("dim_reduction", pymongo.ASCENDING),
            ("reduced_dim", pymongo.ASCENDING),
        ],
        unique=True,
    )
    result_collection.create_index(
        keys=[
            ("config_id", pymongo.ASCENDING),
        ],
        unique=True,
    )

    # Generating & storing model configurations
    configs = generate_configs()
    for config in configs:
        try:
            config_collection.insert_one(config)
        except pymongo.errors.DuplicateKeyError:
            continue

    # Initialising collection to store run results
    configs = config_collection.find({})
    result_docs = []
    for c in configs:
        result_docs.append({"config_id": ObjectId(c["_id"])})
    result_collection.insert_many(result_docs)
