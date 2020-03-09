import torch
import numpy as np
from torch.utils.data import Dataset


def start_idx_dist(x):
    """
    Distribution for sampling the starting index when using data augmentation.
    We favorize the beginning of the sentence.
    
    Arguments:
        x {numpy array} -- Available starting indexes
    
    Returns:
        numpy array -- Probabilities
    """
    dist = [i + 1 for i in range(x)][::-1]
    return np.array(dist) / np.sum(dist)


def treat_tokens(tokens, max_len=512, random_loc=True):
    """
    Treats texts already converted to ids, and performs augmentation.
    Not using tokenization on the fly is faster.
    Note that is function only works for CamemBert (and Roberta).
    It is easily adaptable to Bert by changing the <SOS> and <EOS> indices
    
    Arguments:
        tokens {bert token ids} -- Tokenized sentence
    
    Keyword Arguments:
        max_len {int} -- Maximum token length (default: {512})
        random_loc {bool} -- Whether to consider tokens starting at a random location (default: {True})
    
    Returns:
        list -- Tokens to feed the transformer
    """
    if len(tokens) >= max_len - 2:
        if random_loc:
            loc = np.random.choice(
                np.arange(0, len(tokens) - (max_len - 2)),
                p=start_idx_dist(len(tokens) - (max_len - 2)),
            )
        else:
            loc = 0

        tokens = tokens[loc : loc + max_len - 2]

    tokens = [5] + list(tokens) + [6]
    padding = [0] * (max_len - len(tokens))

    return tokens + padding


def convert_text(text, transformer, max_len=512):
    """
    Adapts a sentence to be fed to a transformer
    
    Arguments:
        text {string} -- Sentence
        transformer {huggingface transformer} -- Transformer to use. It will define the tokenizer.
    
    Keyword Arguments:
        max_len {int} -- Maximum sentence length (default: {512})
    
    Returns:
        list -- Tokens to feed the transformer
    """
    tokens = transformer.tokenizer.tokenize(text)

    if "camembert" in transformer.name:  # CamemBert
        tokens = ["<s>"] + tokens[: max_len - 2] + ["</s>"]
    elif "bert" in transformer.name:  # Bert
        tokens = ["[CLS]"] + tokens[: max_len - 2] + ["[SEP]"]
    else:
        tokens = tokens[:max_len]

    ids = transformer.tokenizer.convert_tokens_to_ids(tokens)
    padding = [0] * (max_len - len(ids))

    return ids + padding


class AltegradTrainDataset(Dataset):
    """
    Torch dataset for training and validating, this one handles text augmentation
    """
    def __init__(self, df_texts, nodes, y, transformer, max_len=512, augment=False):
        """
        Constructor
        
        Arguments:
            df_texts {pandas dataframe} -- Dataframe containing the texts
            nodes {list} -- Nodes to consider
            y {list} -- Labels
            transformer {huggingface transformer} -- Transformer used for tokenization
        
        Keyword Arguments:
            max_len {int} -- Maximum text length (default: {512})
            augment {bool} -- Whether to do the augmentation by selecting a random part of the text (default: {False})
        """
        super().__init__()
        self.max_len = max_len
        self.augment = augment

        self.df_texts = df_texts.iloc[nodes].copy()
        self.df_texts["target"] = y

        tokens = self.df_texts["ids"].values
        tokens = [tokens.split(" ") for tokens in self.df_texts["ids"].values]
        self.tokens = [list(np.array(token).astype(int)) for token in tokens]

        self.y = self.df_texts["target"].values

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        tokens = treat_tokens(self.tokens[idx], self.max_len, random_loc=self.augment)
        return torch.tensor(tokens), torch.tensor(self.y[idx])


class AltegradTestDataset(Dataset):
    """
    Torch dataset for testing and validating
    """
    def __init__(self, df_texts, nodes, transformer, max_len=512):
        """
        Constructor
        
        Arguments:
            df_texts {pandas dataframe} -- Dataframe containing the texts
            nodes {list} -- Nodes to consider
            transformer {huggingface transformer} -- Transformer used for tokenization
        
        Keyword Arguments:
            max_len {int} -- Maximum text length (default: {512})
        """
        super().__init__()
        self.df_texts = df_texts.iloc[nodes].copy()
        texts = list(self.df_texts["treated_text"].astype(str).values)
        self.tokens = np.array([convert_text(text, transformer) for text in texts])

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx]), 0
