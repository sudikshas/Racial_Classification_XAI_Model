import sys
import json
import os

sys.path.insert(0, 'src')

from etl import get_data

def main(targets):
    if ('data' in targets):
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        print(data_cfg)
        get_data(**data_cfg)
    if ('clean' in targets):
        if os.path.exists("data"):
            shutil.rmtree('data/')

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)