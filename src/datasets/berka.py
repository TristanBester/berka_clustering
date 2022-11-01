import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Berka(Dataset):
    def __init__(self, account_path, transaction_path) -> None:
        super().__init__()
        df_acc = pd.read_csv(account_path, sep=";")
        self.account_ids = df_acc.account_id.values

        self.df_trans = pd.read_csv(transaction_path, sep=";", low_memory=False)
        self.df_trans.date = pd.to_datetime(self.df_trans.date, format="%y%m%d")
        # PRIJEM => RECEPTION
        self.df_trans.loc[self.df_trans.type == "PRIJEM", "amount"] = (
            self.df_trans[self.df_trans.type == "PRIJEM"].amount * 1
        )
        # VYDAJ => ISSUE
        self.df_trans.loc[self.df_trans.type == "VYDAJ", "amount"] = (
            self.df_trans[self.df_trans.type == "VYDAJ"].amount * -1
        )

        self._calc_max_len()

    def _calc_max_len(self):
        self.max_len = 0

        for acc_id in self.account_ids:
            df_t = self.df_trans[self.df_trans.account_id == acc_id]

            if df_t.shape[0] > self.max_len:
                self.max_len = df_t.shape[0]

    def __len__(self):
        return len(self.account_ids)

    def __getitem__(self, index):
        account_id = self.account_ids[index]
        amounts = self.df_trans[self.df_trans.account_id == account_id].amount.values

        amounts = (amounts - amounts.mean()) / amounts.std()
        n_pad = self.max_len - len(amounts)
        amounts = np.pad(amounts, n_pad, "constant", constant_values=(0,))[n_pad:]
        amounts = torch.from_numpy(amounts).unsqueeze(0).to(torch.float)
        return amounts, 0
