import re

anime_bbcode = re.compile(r'\[anime=?.*?]')
anime_close_bbcode = re.compile(r'\[/anime]')

manga_bbcode = re.compile(r'\[manga=?.*?]')
manga_close_bbcode = re.compile(r'\[/manga]')

div_bbcode = re.compile(r'\[div=?.*?]')
div_close_bbcode = re.compile(r'\[/div]')

spoiler_bbcode = re.compile(r'\[spoiler=?.*?]')
spoiler_close_bbcode = re.compile(r'\[/spoiler]')

replies_bbcode = re.compile(r'\[replies=[0-9,]+]')
poster_bbcode = re.compile(r'\[poster=[0-9,]+]')

size_bbcode = re.compile(r'\[size=[0-9]+]')
size_close_bbcode = re.compile(r'\[/size]')

profile_bbcode = re.compile(r'\[profile=[0-9]+]')
profile_close_bbcode = re.compile(r'\[/profile]')

person_bbcode = re.compile(r'\[person=[0-9]+]')
person_close_bbcode = re.compile(r'\[/person]')

character_bbcode = re.compile(r'\[character=[0-9]+]')
character_close_bbcode = re.compile(r'\[/character]')

right_bbcode = re.compile(r'\[right]')
right_close_bbcode = re.compile(r'\[/right]')

quote_bbcode = re.compile(r'\[quote]')
quote_close_bbcode = re.compile(r'\[/quote]')

img_bbcode = re.compile(r'\[img]')
img_close_bbcode = re.compile(r'\[/img]')

image_bbcode = re.compile(r'\[image=[0-9]+]')

list_regexps = [
    anime_bbcode,
    anime_close_bbcode,
    manga_bbcode,
    manga_close_bbcode,
    spoiler_bbcode,
    spoiler_close_bbcode,
    replies_bbcode,
    image_bbcode,
    size_bbcode,
    size_close_bbcode,
    character_bbcode,
    character_close_bbcode,
    right_bbcode,
    right_close_bbcode,
    profile_bbcode,
    profile_close_bbcode,
    poster_bbcode,
    img_bbcode,
    img_close_bbcode,
    div_bbcode,
    div_close_bbcode,
    quote_bbcode,
    quote_close_bbcode,
    person_bbcode,
    person_close_bbcode
]
