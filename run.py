#!/usr/bin/env python3

import torch.multiprocessing as mp
from opencompass.cli.main import main

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()

