import torch
import warnings
from datetime import date

warnings.filterwarnings("ignore")

SEED = 2019
MAX_LEN = 512


DATA_PATH = '../input/'
TEXT_PATH = '../input/text/text/'

CP_PATH = f'../checkpoints/{date.today()}/'


CLASSES = ['business/finance', 'education/research', 'entertainment', 'health/medical', 'news/press', 'politics/government/law', 'sports', 'tech/science']
NUM_CLASSES = len(CLASSES)

NUM_WORKERS = 4
VAL_BS = 32

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NB_NODES = 28003