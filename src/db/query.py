def get_results_missing_run(collection, run_name):
    configs = collection.find(
        {
            "$or": [
                {run_name: {"$exists": False}},
            ]
        }
    )
    return configs


def get_config_by_id(collection, id_):
    return collection.find_one({"_id": id_})


def log_run_result(
    collection, run_name, result_id, sc_train, dbi_train, sc_test, dbi_test, train_time
):
    collection.update_one(
        {"_id": result_id},
        {
            "$set": {
                run_name: {
                    "results": {
                        "train": {"SC": sc_train, "DBI": dbi_train},
                        "test": {"SC": sc_test, "DBI": dbi_test},
                    },
                    "train_time": train_time,
                },
            }
        },
    )
