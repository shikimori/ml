import bbcode

from .bb_regexps import list_regexps
from .smileys import smiley_groups


def remove_custom_bbcodes(text: str) -> str:
    for regex in list_regexps:
        text = regex.sub('', text)

    return text


def remove_bbcode(text: str) -> str:
    parser = bbcode.Parser()

    return parser.strip(text)


def remove_shiki_smileys(text: str) -> str:
    for smile_group in smiley_groups:
        for smile in smile_group:
            text = text.replace(smile, '')

    return text


def clean_txt(text: str) -> str:
    text = remove_bbcode(text)
    text = remove_custom_bbcodes(text)
    text = remove_shiki_smileys(text)

    return text
