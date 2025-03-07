# Adapted from: https://github.com/jirifilip/pyARC

import pandas as pd

from src.algorithm.cba.data_structures.transaction_db import TransactionDB


def transactiondb_to_dataframe(transactionDB: TransactionDB):
    new_transaction_list = [
        [item.value for item in transaction.items] + [transaction.class_val.value]
        for transaction in transactionDB.data
    ]

    result = pd.DataFrame(new_transaction_list, columns=transactionDB.header)

    return result
