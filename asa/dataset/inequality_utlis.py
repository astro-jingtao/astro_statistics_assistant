import re

def parse_inequality(inequaliyt_string):
    # deal with ''
    splitted = re.split(r'(<=|>=|<|>|==)', inequaliyt_string.replace(" ", ""))
    return [s for s in splitted if s != '']


def parse_op(string):
    # +, -, *, /, **, (, )
    splitted = re.split(r'(\+|-|\*|/|\*\*|\(|\))', string.replace(" ", ""))
    return [s for s in splitted if s != '']


def parse_and_or(string):
    # ~, &, |, [, ]
    splitted = re.split(r'(~|&|\||\[|\])', string.replace(" ", ""))
    return [s for s in splitted if s != '']


def is_inequality(string):
    return re.search(r'(<=|>=|<|>)', string) is not None