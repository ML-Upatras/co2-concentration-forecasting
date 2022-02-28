# delete folder .darts, results and figures using Pathlib

import shutil
from pathlib import Path

if __name__ == '__main__':
    RESULTS_PATH = Path('results')
    FIGURES_PATH = Path('figures')
    DART_PATH = Path('.darts')

    shutil.rmtree(RESULTS_PATH, ignore_errors=True)
    shutil.rmtree(FIGURES_PATH, ignore_errors=True)
    shutil.rmtree(DART_PATH, ignore_errors=True)
    print('Deleted results, figures and .darts folder')
