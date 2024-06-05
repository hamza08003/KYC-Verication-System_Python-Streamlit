from fuzzywuzzy import fuzz


def normalize_text(text):
    return ' '.join(text.lower().replace(',', '').split())

def compare_names(name1, name2):
    return fuzz.token_set_ratio(normalize_text(name1), normalize_text(name2)) > 85

def compare_addresses(address1, address2):
    return fuzz.partial_ratio(normalize_text(address1), normalize_text(address2)) > 70
