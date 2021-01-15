import json
import os

def get_data(fairface_label_train, fairface_label_val, fairface_pad025, fairface_pad125, outdir):
    """
    Downloads and saves required data into outdir.
    """

    # Checks to see if outdir exists
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Define out paths
    train_csv = os.path.join(outdir, "fairface_label_train.csv")
    val_csv = os.path.join(outdir, "fairface_label_val.csv")

    pad025 = os.path.join(outdir, "fairface_pad025")
    pad125 = os.path.join(outdir, "fairface_pad125")

    # Connect csv files
    os.symlink(fairface_label_train, train_csv)
    os.symlink(fairface_label_val, val_csv)

    os.symlink(fairface_pad025, pad025)
    os.symlink(fairface_pad125, pad125)

    print("Raw data linked")
    return 
