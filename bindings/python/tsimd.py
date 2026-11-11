import os

from tokensim import Error, Tokenizer, load, tokenizer, version

_MODEL_URLS = {
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
}

_DEFAULT_CACHE = os.path.join(
    os.environ.get("XDG_CACHE_HOME",
                   os.path.join(os.path.expanduser("~"), ".cache")),
    "tokensim")


def load_model(name, cache_dir=None):
    """Load a tokenizer by model-family name, like tiktoken's
    get_encoding. The vocabulary is downloaded once on first use
    and cached under cache_dir (default ~/.cache/tokensim), so
    later runs and other processes reuse it offline."""
    import urllib.request

    if name not in _MODEL_URLS:
        raise Error(f"unknown model {name!r}; known: "
                    f"{', '.join(sorted(_MODEL_URLS))}")
    cache_dir = cache_dir or _DEFAULT_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, name + ".tiktoken")
    if not os.path.exists(path):
        url = _MODEL_URLS[name]
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return tokenizer(path, "tiktoken")


