"""The languages the Ghana Voice base model supports.

Each language occupies a fixed id (0-41) that indexes both the language token prepended to
the phoneme sequence and the speaker-embedding slot. These ids MUST match the base model's
training, so do not reorder them.
"""

# id -> (registry_name, iso_639_3, human_name)
LANGUAGES = {
    0:  ("Akuapem_Twi_twi", "twi", "Akuapem Twi"),
    1:  ("Anyin_any", "any", "Anyin"),
    2:  ("Asante_Twi_twi", "twi", "Asante Twi"),
    3:  ("Avatime_avn", "avn", "Avatime"),
    4:  ("Bassar_Ntcham_bud", "bud", "Bassar (Ntcham)"),
    5:  ("Bimoba_bim", "bim", "Bimoba"),
    6:  ("Birifor_Southern_biv", "biv", "Southern Birifor"),
    7:  ("Bissa_bib", "bib", "Bissa"),
    8:  ("Buli_bwu", "bwu", "Buli"),
    9:  ("Chumburung_ncu", "ncu", "Chumburung"),
    10: ("Dagaare_dga", "dga", "Dagaare"),
    11: ("Dagbani_dag", "dag", "Dagbani"),
    12: ("Dangme_ada", "ada", "Dangme"),
    13: ("Deg_mzw", "mzw", "Deg"),
    14: ("Ewe_ewe", "ewe", "Ewe"),
    15: ("Fante_fat", "fat", "Fante"),
    16: ("Fulfulde_Maasina_ffm", "ffm", "Maasina Fulfulde"),
    17: ("Gikyode_acd", "acd", "Gikyode"),
    18: ("Gonja_gjn", "gjn", "Gonja"),
    19: ("Hausa_hau", "hau", "Hausa"),
    20: ("Kabiye_kbp", "kbp", "Kabiye"),
    21: ("Kasem_xsm", "xsm", "Kasem"),
    22: ("Konkomba_xon", "xon", "Konkomba"),
    23: ("Konni_kma", "kma", "Konni"),
    24: ("Kusaal_kus", "kus", "Kusaal"),
    25: ("Lelemi_lef", "lef", "Lelemi"),
    26: ("Mampruli_maw", "maw", "Mampruli"),
    27: ("Nawuri_naw", "naw", "Nawuri"),
    28: ("Ninkare_gur", "gur", "Ninkare (Frafra)"),
    29: ("Nkonya_nko", "nko", "Nkonya"),
    30: ("Ntrubo_ntr", "ntr", "Ntrubo"),
    31: ("Nzema_nzi", "nzi", "Nzema"),
    32: ("Paasaal_sig", "sig", "Paasaal"),
    33: ("Sehwi_sfw", "sfw", "Sehwi"),
    34: ("Sekpele_lip", "lip", "Sekpele"),
    35: ("Selee_snw", "snw", "Selee"),
    36: ("Sisaala_Tumulung_sil", "sil", "Tumulung Sisaala"),
    37: ("Siwu_akp", "akp", "Siwu"),
    38: ("Tampulma_tpm", "tpm", "Tampulma"),
    39: ("Tem_kdh", "kdh", "Tem"),
    40: ("Tuwuli_bov", "bov", "Tuwuli"),
    41: ("Vagla_vag", "vag", "Vagla"),
}

N_LANGUAGES = len(LANGUAGES)

# Lookup helpers: accept an id, the registry name, the iso code, or the human name
# (case-insensitive) so users can specify a language however is convenient.
_ALIASES = {}
for _id, (_name, _iso, _human) in LANGUAGES.items():
    for key in (str(_id), _name, _iso, _human):
        _ALIASES[key.lower()] = _id


def resolve(language):
    """Return the integer language id for an id / registry name / iso code / human name."""
    if isinstance(language, int):
        if language not in LANGUAGES:
            raise KeyError(f"Unknown language id {language} (valid 0-{N_LANGUAGES - 1})")
        return language
    key = str(language).strip().lower()
    if key not in _ALIASES:
        raise KeyError(
            f"Unknown language '{language}'. Use an id (0-{N_LANGUAGES - 1}), iso code, or name. "
            f"See `ghanavoice languages` for the full list."
        )
    return _ALIASES[key]


def name(language):
    return LANGUAGES[resolve(language)][2]


def iso(language):
    return LANGUAGES[resolve(language)][1]


def registry_name(language):
    return LANGUAGES[resolve(language)][0]
