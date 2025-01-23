'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-10-08 15:02:36
FilePath: /InASR/text/build_tokenizer.py
'''
from pathlib import Path
from typing import Iterable, Union

# from typeguard import typechecked

from text.abs_tokenizer import AbsTokenizer
from text.char_tokenizer import CharTokenizer
from text.hugging_face_tokenizer import HuggingFaceTokenizer
from text.sentencepiece_tokenizer import SentencepiecesTokenizer
from text.whisper_tokenizer import OpenAIWhisperTokenizer
from text.word_tokenizer import WordTokenizer


def build_tokenizer(
    token_type: str,
    bpemodel: Union[Path, str, Iterable[str]] = None,
    non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: str = None,
) -> AbsTokenizer:
    """A helper function to instantiate Tokenizer"""
    # assert typechecked()
    if token_type == "bpe":
        if bpemodel is None:
            raise ValueError('bpemodel is required if token_type = "bpe"')

        if remove_non_linguistic_symbols:
            raise RuntimeError(
                "remove_non_linguistic_symbols is not implemented for token_type=bpe"
            )
        return SentencepiecesTokenizer(bpemodel)

    if token_type == "hugging_face":
        if bpemodel is None:
            raise ValueError('bpemodel is required if token_type = "hugging_face"')

        if remove_non_linguistic_symbols:
            raise RuntimeError(
                "remove_non_linguistic_symbols is not "
                + "implemented for token_type=hugging_face"
            )
        return HuggingFaceTokenizer(bpemodel)

    elif token_type == "word":
        if remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            return WordTokenizer(
                delimiter=delimiter,
                non_linguistic_symbols=non_linguistic_symbols,
                remove_non_linguistic_symbols=True,
            )
        else:
            return WordTokenizer(delimiter=delimiter)

    elif token_type == "char":
        return CharTokenizer(
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        )

    elif "whisper" in token_type:
        return OpenAIWhisperTokenizer(bpemodel)

    else:
        raise ValueError(
            f"token_mode must be one of bpe, word, char or phn: " f"{token_type}"
        )
