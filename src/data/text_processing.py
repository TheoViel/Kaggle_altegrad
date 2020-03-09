import re
import html
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


backslashes = "\x10\x08\uf04a\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600\x95\n\ue807\x80\x9d\x92\uf071\uf03d\uf031\uf028\uf02d\uf061\ue014\uf029\x96\x9f\uf020\x9c\x81\uf04c\uf070\uf02f\uf032\uf005\uf818\U0001f92f\U0001f92a\x91\ue602\ue613\x13\uf10a\uf0e0"


def process_in_parallel(function, list_):
    """
    Helper to use multiprocessing
    
    Arguments:
        function {function} -- Function to apply
        list_ {list} -- List to multiprocess
    
    Returns:
        [type] -- Function applied to the list
    """
    with Pool(cpu_count()) as p:
        output = p.map(function, list_)
    return output


def clean_consec_chars(x):
    """
    Removes consecutive characters
    
    Arguments:
        x {string} -- Text to treat
    
    Returns:
        string -- Treated text
    """
    return re.sub(r"([\w\W])\1\1+", r"\1\1\1", x)


def remove_url(x):
    """
    Removes urls
    
    Arguments:
        x {string} -- Text to treat
    
    Returns:
        string -- Treated text
    """
    x = re.sub(r"http\S+", " URL ", x)
    x = re.sub(r"www\S+", " URL ", x)
    return re.sub(r"@\S+", " USERNAME ", x)


def clean_apostrophes(x):
    """
    Cleans apostrophes
    
    Arguments:
        x {string} -- Text to treat
    
    Returns:
        string -- Treated text
    """
    apostrophes = ["’", "‘", "´", "`"]
    for s in apostrophes:
        x = re.sub(s, "'", x)
    return x


def clean_spaces(x):
    """
    Cleans extra spaces

    Arguments:
        x {string} -- Text to treat
    
    Returns:
        string -- Treated text
    """
    x = x.strip()
    x = re.sub("\s+", " ", x)
    x = re.sub("_+", " ", x)
    return x


def remove_too_long(x):
    """
    Removes too long words
    
    Arguments:
        x {string} -- Text to treat
    
    Returns:
        string -- Treated text
    """
    try:
        x = " ".join([w for w in x.split(" ") if len(w) < 25])
    except:
        print(x)
    return x


def remove_alternate(x):
    """
    Removes the "alternate" word 
    
    Arguments:
        x {string} -- Text to treat
    
    Returns:
        string -- Treated text
    """
    x = re.sub("#alternate", "", x)
    x = re.sub("alternate", "", x)
    return x


def clean_numbers(text):
    """
    Cleans numbers
    
    Arguments:
        x {string} -- Text to treat
    
    Returns:
        string -- Treated text
    """

    text = re.sub(r"(\d+)([a-zA-Z])", "\g<1> \g<2>", text)
    text = re.sub(r"(\d+) (th|st|nd|rd) ", "\g<1>\g<2> ", text)
    text = re.sub(r"(\d+),(\d+)", "\g<1>\g<2>", text)
    return text


def treat_texts(texts):
    """
    Apply the specified functions to the texts list using multiprocessing
    
    Arguments:
        x {list of strings} -- Texts to treat
    
    Returns:
        list of strings -- Treated text
    """

    functions = [
        clean_spaces,
        remove_url,
        clean_numbers,
        clean_apostrophes,
        clean_consec_chars,
        remove_alternate,
        remove_too_long,
        clean_spaces,
    ]

    for function in tqdm(functions):
        texts = process_in_parallel(function, texts)

    return texts
