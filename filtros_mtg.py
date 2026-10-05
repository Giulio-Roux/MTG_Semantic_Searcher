"""Filtros por metadados + busca hibrida para o MTG Semantic Searcher.

Ideia: separar a consulta em duas partes.
  1. FILTROS (tipo, cor, custo)  -> codigo comum, garantido, sem ambiguidade.
  2. SIGNIFICADO (o resto)       -> embedding, que ordena as cartas que passaram.

Regra para nao confundir assunto com alvo:
  so as palavras da FRASE NOMINAL INICIAL viram filtro.
  "creature that destroys artifacts"  -> filtro: Creature   | semantico: "destroys artifacts"
  "destroy target creature"           -> sem filtro         | semantico: a frase inteira
  (essa mesma regra vale para subtipos: "dinosaur that costs 7 mana" -> filtro: dinosaur;
   "destroy all dinosaurs" -> sem filtro, pois "dinosaurs" aqui e' o ALVO, nao a propria carta)
"""
import difflib
import json
import re
import numpy as np

# ---------------------------------------------------------------- parametros
# "cheap"/"expensive" NAO viram corte (cmc <= 2): viram uma PREFERENCIA na ordenacao.
# Entre as COST_POOL cartas mais relevantes, o custo entra como um desempate suave.
COST_POOL = 100      # quantas cartas (as mais relevantes) sao reordenadas por custo
COST_WEIGHT = 0.5    # peso do custo: 0 = ignora o custo; 1 = custo pesa tanto quanto a relevancia

# Nome do campo de tags funcionais no seu JSON (ex.: "removal", "ramp", "card-advantage").
# Se o nome real for outro, troque so aqui.
TAG_FIELD = "otags"

TYPE_WORDS = {
    "creature": "creature", "creatures": "creature",
    "instant": "instant", "instants": "instant",
    "sorcery": "sorcery", "sorceries": "sorcery",
    "artifact": "artifact", "artifacts": "artifact",
    "enchantment": "enchantment", "enchantments": "enchantment",
    "land": "land", "lands": "land",
    "planeswalker": "planeswalker", "planeswalkers": "planeswalker",
    "battle": "battle", "battles": "battle",
    "legendary": "legendary",
}
COLOR_WORDS = {"white": "W", "blue": "U", "black": "B", "red": "R", "green": "G"}
FILLERS = {"a", "an", "the", "some"}   # palavras que podem aparecer na frase inicial sem mudar nada

_CMC_PATTERN = re.compile(r"\b(?:cmc|mana value)\s*(<=|>=|=|<|>)?\s*(\d+)\b", re.IGNORECASE)

# Nome de carta entre aspas: "Lightning Bolt", ou 'Lightning Bolt'. Busca unitaria por uma carta
# especifica. Extraido ANTES de tudo, pra numeros/palavras dentro das aspas nao virarem outro filtro.
_NAME_PATTERN = re.compile(r'["\'](.+?)["\']')

_STAT_PATTERN = re.compile(
    r"\b(power|toughness|loyalty|defense)\s*(<=|>=|=|<|>)?\s*(\d+)"
    r"(?:\s*(or\s+(?:greater|more|higher)|or\s+(?:less|lower)))?\b", re.IGNORECASE)

_PRODUCES_PATTERN = re.compile(
    r"\bproduces?\s+((?:(?:white|blue|black|red|green)\s*(?:,|and|or)?\s*)+)mana\b", re.IGNORECASE)

# Indice dinamico de keywords
_KEYWORD_INDEX = {}


def set_keyword_index(cards):
    """Chame uma vez, depois de carregar `cards`, para habilitar o filtro por keyword
    ("flying creature", "creature with first strike and trample").
    Sem chamar isso, filtros de keyword simplesmente nao sao reconhecidos (nao quebra nada)."""
    global _KEYWORD_INDEX
    idx = {}
    for c in cards:
        for kw in c.get("keywords") or []:
            idx.setdefault(kw.strip().lower(), kw.strip())
    _KEYWORD_INDEX = idx
    return idx


# Indice dinamico de TAGS funcionais ("removal", "ramp", "card-advantage")
_TAG_INDEX = {}


def set_tag_index(cards):
    """Chame uma vez, depois de carregar `cards`, para habilitar o filtro por tag
    ("removal creature", "green ramp"). Sem chamar isso, o filtro simplesmente nao e' reconhecido."""
    global _TAG_INDEX
    idx = {}
    for c in cards:
        for t in c.get(TAG_FIELD) or []:
            idx.setdefault(t.strip().lower(), t.strip())
    _TAG_INDEX = idx
    return idx


# =====================================================================================
# Indice dinamico de SUPERTIPOS/TIPOS/SUBTIPOS (Legendary, Aura, Dinosaur, Equipment,
# Rabbit, Vehicle, Saga, Desert, ...), no mesmo espirito de _KEYWORD_INDEX / _TAG_INDEX acima
#
# type_line vem tipicamente como "Legendary Enchantment — Aura" ou "Creature — Dinosaur Warrior":
# a extracao so tira o travessao (em-dash "—", e tambem " - " por seguranca) e pega cada palavra que sobra.
# =====================================================================================
_TYPE_LINE_SPLIT = re.compile(r"\s*(?:—|--|-)\s*")   # em-dash "—", "--" ou " - " entre tipo e subtipo
_TYPE_INDEX = {}


def set_type_index(cards):
    """Chame uma vez, depois de carregar `cards`, para habilitar o reconhecimento de
    supertipos/subtipos que NAO estao na lista fixa TYPE_WORDS (ex.: 'dinosaur', 'aura',
    'equipment', 'vehicle', 'rabbit', 'saga', 'legendary'...). Nao troca nada do que ja existe:
    TYPE_WORDS continua sendo checado primeiro; isso so cobre o que TYPE_WORDS nao cobre.
    Sem chamar isso, so os tipos principais de TYPE_WORDS continuam sendo reconhecidos
    (nao quebra nada do comportamento atual)."""
    global _TYPE_INDEX
    idx = {}
    for c in cards:
        type_line = c.get("type_line") or ""
        for metade in _TYPE_LINE_SPLIT.split(type_line):
            for palavra in metade.split():
                limpa = re.sub(r"[^\w]", "", palavra)
                if limpa:
                    idx.setdefault(limpa.lower(), limpa)
    _TYPE_INDEX = idx
    return idx


def set_metadata_indexes(cards):
    """Atalho: chama set_keyword_index e set_tag_index de una vez."""
    return set_keyword_index(cards), set_tag_index(cards)


def set_all_indexes(cards):
    """atalho equivalente a set_metadata_indexes, mas tambem chamando set_type_index.
    Usar este no lugar de set_metadata_indexes se quiser habilitar o filtro por
    subtipo/supertipo (aura, dinosaur, equipment...) junto com keywords e tags."""
    return set_keyword_index(cards), set_tag_index(cards), set_type_index(cards)


def _phrases_by_length(index):
    """Frases ordenadas da mais longa para a mais curta (em palavras), para o regex tentar
    casar uma frase de varias palavras (ex.: 'double strike', 'card draw') antes de um prefixo parcial."""
    return sorted(index.keys(), key=lambda p: -len(p.split()))


def _extract_phrases(text, index, connectors=("with", "having", "has")):
    """Acha ocorrencias conhecidas (de `index`) em qualquer lugar do texto, remove do texto
    e devolve (texto_restante, [nomes_canonicos]). Usada tanto para keywords quanto para tags."""
    if not index:
        return text, []
    found = []
    for phrase in _phrases_by_length(index):
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            found.append(index[phrase])
            text = pattern.sub(" ", text)
    if found and connectors:
        # "with"/"having"/"has" perdem sentido depois que o argumento delas foi removido
        text = re.sub(r"\b(" + "|".join(connectors) + r")\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*,\s*|\s+and\s+", " ", text)   # sobras de listas: "flying and trample" -> "  "
    return text, found


def _extract_keywords(text):
    return _extract_phrases(text, _KEYWORD_INDEX)


def _extract_tags(text):
    return _extract_phrases(text, _TAG_INDEX)


def _type_index_lookup(word):
    """Procura `word` (ja' minuscula/sem pontuacao) no indice dinamico de tipos/subtipos.
    Tenta a palavra exata e, se nao achar, a forma singular removendo um 's' final
    ("dinosaurs" -> "dinosaur"), ja' que na consulta a palavra pode vir no plural mesmo
    que no type_line ela sempre apareca no singular. Devolve o nome canonico (com a grafia
    do Scryfall) ou None."""
    if word in _TYPE_INDEX:
        return _TYPE_INDEX[word]
    if word.endswith("s") and word[:-1] in _TYPE_INDEX:
        return _TYPE_INDEX[word[:-1]]
    return None


# ---------------------------------------------------------- interpretar a consulta
def parse_query(query: str):
    """Devolve (filtros, texto_semantico)."""
    filters = {"types": [], "colors": [], "colorless": False,
               "cmc_min": None, "cmc_max": None, "cost_pref": None,
               "power_min": None, "power_max": None, "toughness_min": None, "toughness_max": None,
               "loyalty_min": None, "loyalty_max": None, "defense_min": None, "defense_max": None,
               "keywords": [], "produced_mana": [], "tags": [], "name": None}

    # 0) nome entre aspas, em qualquer lugar da consulta: "Lightning Bolt"
    m = _NAME_PATTERN.search(query)
    if m:
        filters["name"] = m.group(1).strip()
        query = _NAME_PATTERN.sub(" ", query, count=1)

    # 1) custo explicito em qualquer parte: "cmc 3", "cmc <= 2", "mana value >= 5"
    def _cmc(m):
        op, n = m.group(1) or "=", int(m.group(2))
        if op == "=":
            filters["cmc_min"] = filters["cmc_max"] = n
        elif op == "<=":
            filters["cmc_max"] = n
        elif op == "<":
            filters["cmc_max"] = n - 1
        elif op == ">=":
            filters["cmc_min"] = n
        elif op == ">":
            filters["cmc_min"] = n + 1
        return " "
    rest = _CMC_PATTERN.sub(_cmc, query).strip()

    # 1b) keywords em qualquer lugar da consulta ("flying creature", "creature with trample and haste")
    rest, kws = _extract_keywords(rest)
    filters["keywords"] = kws

    rest, tags = _extract_tags(rest)
    filters["tags"] = tags

    # 1c) "produces green mana" / "produces red and white mana" (para terrenos/rochas de mana)
    def _produces(m):
        cols = [COLOR_WORDS[w] for w in re.findall(r"white|blue|black|red|green", m.group(1), re.IGNORECASE)]
        for c in cols:
            if c not in filters["produced_mana"]:
                filters["produced_mana"].append(c)
        return " "
    rest = _PRODUCES_PATTERN.sub(_produces, rest).strip()

    # 1d) power/toughness/loyalty/defense: "power >= 3", "toughness 1", "loyalty 5 or greater"
    def _stat(m):
        field, op, n, phrase = m.group(1).lower(), m.group(2), int(m.group(3)), (m.group(4) or "").lower()
        if phrase.startswith("or") and "great" not in phrase and "more" not in phrase and "high" not in phrase:
            op = "<="                      # "... or less"/"or lower"
        elif phrase:
            op = ">="                      # "... or greater"/"or more"/"or higher"
        op = op or "="
        if op == "=":
            filters[f"{field}_min"] = filters[f"{field}_max"] = n
        elif op == "<=":
            filters[f"{field}_max"] = n
        elif op == "<":
            filters[f"{field}_max"] = n - 1
        elif op == ">=":
            filters[f"{field}_min"] = n
        elif op == ">":
            filters[f"{field}_min"] = n + 1
        return " "
    rest = _STAT_PATTERN.sub(_stat, rest).strip()

    # 2) frase nominal inicial: consome palavras conhecidas ate a primeira desconhecida
    tokens = rest.split()
    consumed, recognized = 0, False
    for tok in tokens:
        w = re.sub(r"[^\w]", "", tok.lower())
        if w in FILLERS:
            consumed += 1
        elif w in TYPE_WORDS:
            if TYPE_WORDS[w] not in filters["types"]:
                filters["types"].append(TYPE_WORDS[w])
            consumed += 1
            recognized = True
        elif _type_index_lookup(w) is not None:
            canonico = _type_index_lookup(w).lower()
            if canonico not in filters["types"]:
                filters["types"].append(canonico)
            consumed += 1
            recognized = True
        elif w in COLOR_WORDS:
            if COLOR_WORDS[w] not in filters["colors"]:
                filters["colors"].append(COLOR_WORDS[w])
            consumed += 1
            recognized = True
        elif w == "colorless":
            filters["colorless"] = True
            consumed += 1
            recognized = True
        elif w == "cheap":
            filters["cost_pref"] = "cheap"
            consumed += 1
            recognized = True
        elif w == "expensive":
            filters["cost_pref"] = "expensive"
            consumed += 1
            recognized = True
        else:
            break

    had_cmc = filters["cmc_min"] is not None or filters["cmc_max"] is not None
    if not recognized:
        consumed = 0                      # nada reconhecido: nao remove palavra nenhuma
    semantic = " ".join(tokens[consumed:])
    semantic = re.sub(r"^(that|which|who)\s+", "", semantic, flags=re.IGNORECASE).strip(" ,;.")

    # Se sobrou nada (ex.: "cheap green creature"), semantic = "": a consulta e' so filtros,
    # e a busca nao deve fingir que ha um "significado" a procurar.
    return filters, semantic


# ------------------------------------------------------ metadados de uma carta
def _first_mana_cost(card: dict) -> str:
    mc = card.get("mana_cost")
    if not mc and card.get("faces"):
        mc = card["faces"][0].get("mana_cost")
    return (mc or "").split(" // ")[0]


def card_cmc(card: dict) -> float:
    mana_cost = card.get("mana_cost") or ""          # "or": protege contra mana_cost = null

    if not mana_cost and card.get("faces"):
        mana_cost = card["faces"][0].get("mana_cost") or ""

    total = 0

    for symbol in re.findall(r"\{([^}]+)\}", mana_cost):
        if symbol.isdigit():
            total += int(symbol)
        elif symbol in {"X", "Y", "Z"}:
            continue
        else:
            total += 1

    return total


def _parse_stat(value):
    """power/toughness/loyalty/defense podem ser '*', '1+*', 'X', None, etc.
    So retorna um numero quando da' para comparar com seguranca; senao, None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def has_cost(card: dict) -> bool:
    """Tem custo de mana impresso, ou e' terreno (terreno tem custo 0 de verdade).
    Cartas sem custo e que nao sao terreno (ex.: resultado de 'meld', faces que nao se conjuram)
    aparecem como cmc 0, mas nao sao 'baratas': so nao podem ser conjuradas pelo custo."""
    return bool(_first_mana_cost(card).strip()) or "land" in (card.get("type_line") or "").lower()


def card_colors(card: dict) -> set:
    colors = card.get("colors")
    if colors is None:                    # cartas de dupla face podem vir sem 'colors'
        symbols = re.findall(r"\{([^}]+)\}", _first_mana_cost(card))
        colors = [c for s in symbols for c in s.split("/") if c in "WUBRG" and len(c) == 1]
    return set(colors)


def _name_parts(card: dict):
    """O nome inteiro e, se for carta de duas faces ('Fire // Ice'), cada metade tambem."""
    name = card.get("name") or ""
    parts = [name]
    if " // " in name:
        parts += name.split(" // ")
    return parts


def _oracle_text_full(card: dict) -> str:
    """oracle_text da carta; se vier vazio mas houver faces (dupla face), junta o texto de
    cada face -- algumas cartas de dupla face nao tem oracle_text no nivel principal."""
    text = card.get("oracle_text")
    if not text and card.get("faces"):
        text = " // ".join(fc.get("oracle_text") or "" for fc in card["faces"])
    return text or ""


def matches(card: dict, f: dict) -> bool:
    if f.get("name"):
        target = f["name"].lower()
        if not any(target in p.lower() for p in _name_parts(card)):
            return False
    type_line = (card.get("type_line") or "").lower()
    # match por PALAVRA inteira (nao substring): substring deixaria "t:land" bater em
    # "Island" so' porque "land" aparece dentro de "Island" como sequencia de letras.
    type_words = set(re.findall(r"[a-z]+", type_line))
    if any(t not in type_words for t in f["types"]):
        return False
    colors = card_colors(card)
    if f["colorless"] and colors:
        return False
    if f["colors"] and not set(f["colors"]).issubset(colors):   # contem TODAS as cores pedidas
        return False
    if f["cmc_min"] is not None or f["cmc_max"] is not None:
        cmc = card_cmc(card)
        if f["cmc_min"] is not None and cmc < f["cmc_min"]:
            return False
        if f["cmc_max"] is not None and cmc > f["cmc_max"]:
            return False
    if f.get("keywords"):
        if not set(f["keywords"]).issubset(set(card.get("keywords") or [])):
            return False
    if f.get("produced_mana"):
        if not set(f["produced_mana"]).issubset(set(card.get("produced_mana") or [])):
            return False
    if f.get("tags"):
        if not set(f["tags"]).issubset(set(card.get(TAG_FIELD) or [])):
            return False
    if f.get("oracle_words"):
        text_words = set(re.findall(r"[a-z']+", _oracle_text_full(card).lower()))
        if not set(f["oracle_words"]).issubset(text_words):
            return False
    for stat in ("power", "toughness", "loyalty", "defense"):
        lo, hi = f.get(f"{stat}_min"), f.get(f"{stat}_max")
        if lo is not None or hi is not None:
            val = _parse_stat(card.get(stat))
            if val is None:                    # "*", "1+*", ausente etc.: nao da' pra confirmar -> exclui
                return False
            if lo is not None and val < lo:
                return False
            if hi is not None and val > hi:
                return False
    return True


# ------------------------------------------------- parser com LLM (IlumA)
ALLOWED_TYPES = set(TYPE_WORDS.values())
ALLOWED_COLORS = set("WUBRG")

LLM_SYSTEM = """You convert search queries for Magic: The Gathering cards into JSON.

Return ONLY valid JSON, no markdown fences, with exactly these keys:
{"types": [...], "colors": [...], "colorless": false, "cmc_min": null, "cmc_max": null, "cost_pref": null,
 "power_min": null, "power_max": null, "toughness_min": null, "toughness_max": null,
 "loyalty_min": null, "loyalty_max": null, "defense_min": null, "defense_max": null,
 "keywords": [...], "produced_mana": [...], "tags": [...], "name": null, "semantic": "..."}

Rules:
- "types": card types AND subtypes/supertypes that the CARD ITSELF has and that the query
  EXPLICITLY names. This includes: main types (creature, instant, sorcery, artifact, enchantment,
  land, planeswalker, battle), supertypes (legendary, basic, snow, world), and any subtype
  (creature types such as dinosaur/zombie/angel/rabbit/vampire; artifact subtypes such as
  equipment/vehicle/food/clue; enchantment subtypes such as aura/saga/class; land subtypes such
  as desert/cave/gate; and so on for any other permanent type). A type or subtype that is only the
  TARGET of the card's effect does NOT go in "types" -- follow the exact same self-vs-target logic
  for subtypes as for main types: "dinosaur that costs 7 mana" -> types includes "dinosaur"
  (the card itself is a dinosaur); "destroy all dinosaurs" -> types stays empty and "dinosaurs"
  stays in "semantic" (here the dinosaurs are what the card affects, not what it is).
  Include a type/subtype ONLY if that exact word (or its plain singular form) appears literally in
  the query. Do NOT infer a main type from a functional tag ("removal" does not imply
  instant/sorcery), and do NOT infer "creature" just because a creature subtype was named --
  naming the subtype alone (e.g. "dinosaur") is enough, the subtype already narrows the search.
  Write every entry in "types" in lowercase, singular (e.g. "dinosaur", not "Dinosaurs").
  If you are not reasonably sure a word names a real Magic type or subtype, do NOT put it in
  "types" -- leave it in "semantic" instead. A downstream check silently discards anything in
  "types" that isn't a real type/subtype found in the card database, so under-including here is
  always safer than guessing wrong.
- In English noun compounds the LAST noun is the head: "creature destroyer artifact" is an
  ARTIFACT that destroys creatures; "artifact destroyer creature" is a CREATURE that destroys artifacts.
- "colors": colors the card must have, as letters W U B R G. "colorless": true only if asked.
- "cost_pref": "cheap" or "expensive" if the query asks for it, else null. Do NOT turn it into numbers.
- Only EXPLICIT costs ("cmc <= 3", "mana value 2", "costs zero") go into cmc_min/cmc_max.
- "power_min/max", "toughness_min/max": from phrases like "power 4 or greater" (power_min=4),
  "toughness 1 or less" (toughness_max=1), "power exactly 2" (power_min=power_max=2).
- "loyalty_min/max": for planeswalkers, same pattern as power/toughness.
- "defense_min/max": for battles, same pattern.
- "keywords": ability keywords the CARD ITSELF has (e.g. "Flying", "Trample", "First strike",
  "Indestructible", "Deathtouch", "Lifelink", "Haste", "Vigilance", "Menace", "Reach", "Hexproof",
  "Ward", "Double strike", "Flash", "Defender"). Use the exact capitalization shown here. A keyword
  the effect GIVES to something else, or that only appears as reminder text explaining a different
  ability, does NOT count.
- "produced_mana": colors a land/mana-rock PRODUCES (e.g. "land that makes green mana" -> ["G"]).
- "tags": functional category tags for the card (e.g. "removal", "ramp", "card-advantage").
  Only use a tag if the query is clearly asking for that CATEGORY of card, not describing a specific effect.
- "name": set ONLY when the query is asking for one SPECIFIC named card (e.g. quoted text, or "the card
  Lightning Bolt", or just a card name with no description of an effect). Use the name as written. Do NOT
  set "name" when the query describes what a card does in general terms.
- "semantic": English phrase describing what the card DOES, without the words used for filters.
  If the query is ONLY filters (no effect described), use an empty string."""

def _ex(types=(), colors=(), cmc_min=None, cmc_max=None, cost_pref=None,
       power_min=None, power_max=None, toughness_min=None, toughness_max=None,
       loyalty_min=None, loyalty_max=None, defense_min=None, defense_max=None,
       keywords=(), produced_mana=(), tags=(), name=None, semantic=""):
    return {"types": list(types), "colors": list(colors), "colorless": False,
            "cmc_min": cmc_min, "cmc_max": cmc_max, "cost_pref": cost_pref,
            "power_min": power_min, "power_max": power_max,
            "toughness_min": toughness_min, "toughness_max": toughness_max,
            "loyalty_min": loyalty_min, "loyalty_max": loyalty_max,
            "defense_min": defense_min, "defense_max": defense_max,
            "keywords": list(keywords), "produced_mana": list(produced_mana), "tags": list(tags),
            "name": name, "semantic": semantic}


_LLM_EXAMPLES = [
    ("creature that destroys artifacts", _ex(["creature"], semantic="destroys artifacts")),
    ("creature destroyer artifact", _ex(["artifact"], semantic="destroys creatures")),
    ("cheap green creature with flying", _ex(["creature"], ["G"], cost_pref="cheap", semantic="flying")),
    ("destroy target creature", _ex(semantic="destroy target creature")),
    ("cheap green creature", _ex(["creature"], ["G"], cost_pref="cheap", semantic="")),
    ("instant that counters a spell, costs three mana", _ex(["instant"], cmc_max=3, semantic="counters a spell")),
    ("creature with power 4 or greater", _ex(["creature"], power_min=4, semantic="")),
    ("creature with flying and trample", _ex(["creature"], keywords=["Flying", "Trample"], semantic="")),
    ("indestructible green creature", _ex(["creature"], ["G"], keywords=["Indestructible"], semantic="")),
    ("land that produces green mana", _ex(["land"], produced_mana=["G"], semantic="")),
    ("planeswalker with loyalty 6 or higher that draws cards",
     _ex(["planeswalker"], loyalty_min=6, semantic="draws cards")),
    ("cheap green removal", _ex(["instant", "sorcery"], ["G"], cost_pref="cheap", tags=["removal"], semantic="")),
    ('"Lightning Bolt"', _ex(name="Lightning Bolt", semantic="")),
    ("the card Sol Ring", _ex(name="Sol Ring", semantic="")),
    ("removal that generates card advantage", _ex(tags=["removal"], semantic="generates card-advantage")),
    # NOTA: este exemplo mudou. Antes o comentario da regra de "types" dizia que filtro de
    # subtipo nao estava implementado, entao aqui "dinosaur" ficava so no semantic. Agora que
    # subtipo virou filtro de verdade, deixar o exemplo antigo (sem "dinosaur" em types) ensinaria
    # o modelo a fazer exatamente o contrario do que a regra nova pede -- por isso troquei este
    # exemplo em vez de so acrescentar um novo do lado.
    ("dinosaur that costs 7 mana", _ex(types=["dinosaur"], cmc_min=7, cmc_max=7, semantic="")),
    # NOVO: dois exemplos para reforcar subtipo-como-filtro e a distincao "propria carta" x "alvo".
    ("equipment that gives +2/+2", _ex(["equipment"], semantic="gives +2/+2")),
    ("destroy all dinosaurs", _ex(semantic="destroy all dinosaurs")),
]


def _llm_messages(query):
    msgs = [{"role": "system", "content": LLM_SYSTEM}]
    for q, answer in _LLM_EXAMPLES:                      # few-shot: pares user/assistant escritos por nos
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": json.dumps(answer)})
    msgs.append({"role": "user", "content": query})
    return msgs


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    return t


def _as_int(x):
    return x if isinstance(x, int) and not isinstance(x, bool) else None


def _validate(data: dict, query: str):
    """Aceita so valores permitidos: o que o LLM inventar fora das listas (ou fora do indice
    de keywords/tipos do corpus) e descartado."""
    # Aceita qualquer subtipo/supertipo que exista de verdade no indice dinamico _TYPE_INDEX (construido por
    # set_type_index). Sem chamar set_type_index, _TYPE_INDEX fica vazio e o comportamento
    # e' identico ao de antes (so os tipos de ALLOWED_TYPES passam).
    types = [t.lower() for t in data.get("types", [])
             if isinstance(t, str) and (t.lower() in ALLOWED_TYPES or t.lower() in _TYPE_INDEX)]
    colors = [c.upper() for c in data.get("colors", []) if isinstance(c, str) and c.upper() in ALLOWED_COLORS]
    keywords = [_KEYWORD_INDEX[k.lower()] for k in data.get("keywords", [])
                if isinstance(k, str) and k.lower() in _KEYWORD_INDEX]
    produced = [c.upper() for c in data.get("produced_mana", []) if isinstance(c, str) and c.upper() in ALLOWED_COLORS]
    tags = [_TAG_INDEX[t.lower()] for t in data.get("tags", []) if isinstance(t, str) and t.lower() in _TAG_INDEX]
    name = data.get("name")
    name = name.strip() if isinstance(name, str) and name.strip() else None
    semantic = data.get("semantic")
    if not isinstance(semantic, str):
        semantic = ""
    filters = {"types": list(dict.fromkeys(types)), "colors": list(dict.fromkeys(colors)),
               "colorless": data.get("colorless") is True,
               "cmc_min": _as_int(data.get("cmc_min")), "cmc_max": _as_int(data.get("cmc_max")),
               "cost_pref": data.get("cost_pref") if data.get("cost_pref") in ("cheap", "expensive") else None,
               "power_min": _as_int(data.get("power_min")), "power_max": _as_int(data.get("power_max")),
               "toughness_min": _as_int(data.get("toughness_min")), "toughness_max": _as_int(data.get("toughness_max")),
               "loyalty_min": _as_int(data.get("loyalty_min")), "loyalty_max": _as_int(data.get("loyalty_max")),
               "defense_min": _as_int(data.get("defense_min")), "defense_max": _as_int(data.get("defense_max")),
               "keywords": list(dict.fromkeys(keywords)), "produced_mana": list(dict.fromkeys(produced)),
               "tags": list(dict.fromkeys(tags)), "name": name}
    return filters, semantic.strip()


_llm_cache = {}


def parse_query_llm(query, client, model="iluma", temperature=0.8, fallback=parse_query):
    """Interpreta a consulta com o LLM. Se falhar (timeout, JSON ruim...), usa o parser de regras.
    temperature=0.5: abaixo disso a IlumA trava (ver material da aula)."""
    key = query.strip().lower()
    if key in _llm_cache:                                # mesma consulta = nao gasta cota de novo
        return _llm_cache[key]
    try:
        r = client.chat.completions.create(model=model, messages=_llm_messages(query),
                                           temperature=temperature)
        result = _validate(json.loads(_strip_fences(r.choices[0].message.content)), query)
    except Exception as e:                               # nao cacheia falhas: podem ser passageiras
        import traceback
        traceback.print_exc()          # <- mostra o stack completo, com a linha exata que estourou
        print(f"(LLM falhou: {type(e).__name__}; usando o parser de regras)")
        return fallback(query)
    _llm_cache[key] = result
    return result


# --------------------------------------------- parser por sintaxe (sem LLM, sem IA)
# Alternativa 100% deterministica a parse_query_llm: em vez de tentar interpretar linguagem
# natural, o usuario escreve os filtros explicitamente como `prefixo:valor`. Serve de parser
# principal pra quem quer controle total sem depender de API/IlumA, ou de callback caso a
# LLM esteja fora do ar.
#
# Prefixos suportados (todos aceitam `prefixo:valor`; os numericos tambem aceitam
# `<=  >=  <  >  =`):
#   o:palavra   oracle_text contem essa palavra inteira (case-insensitive). Repita o: pra
#               mais de uma palavra -- todas precisam aparecer (AND). ex.: "o:destroys o:artifact"
#   t:tipo      tipo OU subtipo/supertipo (creature, artifact, dinosaur, aura, equipment,
#               legendary...), match de palavra inteira no type_line. Repita t: pra mais de
#               um tipo (AND). ex.: "t:creature t:dinosaur"
#   n:(Nome)    nome exato da carta (ou de uma das faces). Precisa de parenteses por causa
#               dos espacos. ex.: "n:(Lightning Bolt)"
#   c:wubrg     cores que a carta tem, uma letra por cor sem espaco (precisa ter TODAS as
#               letras pedidas). c:colorless == sem nenhuma cor. ex.: "c:rg", "c:colorless"
#   cmc:N / cmc>=N / cmc<=N / cmc>N / cmc<N     custo de mana (mv: e sinonimo de cmc:)
#   pow:N / tou:N / loy:N / def:N               mesmos operadores, pra
#               power / toughness / loyalty / defense
#   k:keyword   keyword de habilidade (Flying, Trample...), match exato contra o indice de
#               keywords do corpus. Repita k: pra mais de uma (AND).
#   pm:wubrg    cores de mana que a carta PRODUZ (terrenos/rochas de mana)
#   tag:tag     tag funcional (removal, ramp...), match exato contra o indice de tags do
#               corpus. Repita tag: pra mais de uma (AND).
#   cost:cheap / cost:expensive   mesma preferencia de desempate por custo do parse_query
#               (nao e' um corte rigido, so' entra no desempate da ordenacao)
#
# Qualquer palavra que NAO fizer parte de um token reconhecido vira texto semantico pro
# embedding, igual nos outros parsers. Valor que nao existir de verdade no corpus (tipo,
# keyword ou tag desconhecidos) e' descartado silenciosamente, mesma politica do _validate.
_SYNTAX_KEYS = ("o", "t", "n", "c", "cmc", "mv", "pow", "tou", "loy", "def",
                "k", "pm", "tag", "cost")
_SYNTAX_TOKEN = re.compile(
    r"(?<!\S)(" + "|".join(_SYNTAX_KEYS) + r")(:|<=|>=|=|<|>)(?:\(([^)]*)\)|(\S+))",
    re.IGNORECASE)
_SYNTAX_STAT_FIELD = {"cmc": "cmc", "mv": "cmc", "pow": "power", "tou": "toughness",
                       "loy": "loyalty", "def": "defense"}


def _syntax_words(raw):
    """Divide um valor (bruto, de dentro ou fora de parenteses) em palavras minusculas,
    sem pontuacao de borda -- usado por o:/t:/k:/tag: quando o valor vem entre parenteses
    (ex.: "o:(deals damage)" filtra as duas palavras, ambas obrigatorias). Mantem hifen
    dentro da palavra (varias tags/keywords usam hifen, ex. "card-advantage")."""
    return [w for w in re.sub(r"[^\w\s-]", " ", raw.lower()).split() if w]


def parse_query_syntax(query: str):
    """Parser por sintaxe explicita (`o:`, `t:`, `n:(...)`, `c:`, `cmc:`, `pow:`, `k:`, `pm:`,
    `tag:`, `cost:` -- ver comentario acima da lista completa). Devolve (filtros, texto
    semantico), no mesmo formato de parse_query/parse_query_llm, entao funciona direto como
    `parser=parse_query_syntax` no hybrid_search."""
    filters = {"types": [], "colors": [], "colorless": False,
               "cmc_min": None, "cmc_max": None, "cost_pref": None,
               "power_min": None, "power_max": None, "toughness_min": None, "toughness_max": None,
               "loyalty_min": None, "loyalty_max": None, "defense_min": None, "defense_max": None,
               "keywords": [], "produced_mana": [], "tags": [], "name": None, "oracle_words": []}

    def _apply_stat(field, op, n):
        if op in (":", "="):
            filters[f"{field}_min"] = filters[f"{field}_max"] = n
        elif op == "<=":
            filters[f"{field}_max"] = n
        elif op == "<":
            filters[f"{field}_max"] = n - 1
        elif op == ">=":
            filters[f"{field}_min"] = n
        elif op == ">":
            filters[f"{field}_min"] = n + 1

    def _token(m):
        key, op, paren, bare = m.group(1).lower(), m.group(2), m.group(3), m.group(4)
        valor = paren if paren is not None else (bare or "").strip(".,;")

        if key == "o":
            for w in _syntax_words(valor):
                if w not in filters["oracle_words"]:
                    filters["oracle_words"].append(w)
        elif key == "t":
            for w in _syntax_words(valor):
                canonico = TYPE_WORDS.get(w) or _type_index_lookup(w)
                if canonico:
                    c = canonico.lower()
                    if c not in filters["types"]:
                        filters["types"].append(c)
        elif key == "n":
            filters["name"] = (paren if paren is not None else bare or "").strip()
        elif key == "c":
            if valor.lower() == "colorless":
                filters["colorless"] = True
            else:
                for ch in valor.upper():
                    if ch in ALLOWED_COLORS and ch not in filters["colors"]:
                        filters["colors"].append(ch)
        elif key in _SYNTAX_STAT_FIELD and valor.lstrip("-").isdigit():
            _apply_stat(_SYNTAX_STAT_FIELD[key], op, int(valor))
        elif key == "k":
            for w in _syntax_words(valor):
                canonico = _KEYWORD_INDEX.get(w)
                if canonico and canonico not in filters["keywords"]:
                    filters["keywords"].append(canonico)
        elif key == "pm":
            for ch in valor.upper():
                if ch in ALLOWED_COLORS and ch not in filters["produced_mana"]:
                    filters["produced_mana"].append(ch)
        elif key == "tag":
            for w in _syntax_words(valor):
                canonico = _TAG_INDEX.get(w)
                if canonico and canonico not in filters["tags"]:
                    filters["tags"].append(canonico)
        elif key == "cost" and valor.lower() in ("cheap", "expensive"):
            filters["cost_pref"] = valor.lower()
        return " "

    semantic = _SYNTAX_TOKEN.sub(_token, query)
    semantic = re.sub(r"\s+", " ", semantic).strip(" ,;.")
    return filters, semantic


# ------------------------------------------------------------- busca hibrida
def _name_rank(card: dict, query_name: str):
    """Ordena cartas que ja passaram no filtro de nome: exato primeiro, depois prefixo,
    depois substring generico; empate por nome mais curto (mais 'especifico') e ordem alfabetica."""
    q = query_name.lower()
    best = (2, len(card.get("name") or ""), card.get("name") or "")
    for p in _name_parts(card):
        pl = p.lower()
        if pl == q:
            return (0, len(p), p)
        if pl.startswith(q):
            best = min(best, (1, len(p), p))
    return best


def find_card(cards, query, k=5, fuzzy_threshold=0.6):
    """Busca RAPIDA por nome (sem embedding, sem IlumA) -- para achar UMA carta especifica,
    inclusive com erro de digitacao ou nome parcial. Prioridade:
    1) nome exato (case-insensitive)  2) comeca com a consulta  3) consulta e' substring do nome
    4) parecido o suficiente (difflib), quando nada exato/prefixo/substring foi encontrado.
    Devolve ate k cartas (dicts originais de `cards`), da melhor pra pior."""
    q = query.strip().lower()
    if not q:
        return []
    exact, prefix, substring = [], [], []
    for c in cards:
        for p in _name_parts(c):
            pl = p.lower()
            if pl == q:
                exact.append(c); break
            elif pl.startswith(q):
                prefix.append(c); break
            elif q in pl:
                substring.append(c); break
    ordered = exact + prefix + substring
    if ordered:
        return ordered[:k]
    # nada bateu por substring: tenta por semelhanca (tolera erro de digitacao)
    scored = []
    for c in cards:
        ratio = max(difflib.SequenceMatcher(None, q, p.lower()).ratio() for p in _name_parts(c))
        if ratio >= fuzzy_threshold:
            scored.append((ratio, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]


def _z(x):
    """Padroniza (media 0, desvio 1) para poder somar relevancia e custo na mesma escala."""
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def hybrid_search(query, model, embeddings, cards, query_prefix, k=10, verbose=True, parser=parse_query,
                  embed_full_query=True, alpha_if_no_semantic=False):
    """cards: lista de dicts ALINHADA com as linhas de `embeddings` (mesma ordem).
    Com "cheap"/"expensive" na consulta, o campo 'score' vira uma nota combinada
    (relevancia + custo); a similaridade pura fica em 'sim'.
    Consulta so de filtros + cheap/expensive (ex.: "cheap green creature"): nao ha relevancia
    a preservar, entao ordena direto por custo (desempate pela similaridade).
    embed_full_query=True (padrao): embeda a consulta INTEIRA, mesmo depois de tirar os filtros;
    False: embeda so o texto que sobrou (ex.: 'destroys artifacts').
    alpha_if_no_semantic=False (padrao): se True e a consulta for SO filtro (nada sobrou de
    texto semantico e sem preferencia cheap/expensive), pula o modelo de embedding de vez e
    devolve os resultados em ordem alfabetica pelo nome -- nao ha "relevancia" a calcular
    quando a consulta inteira ja virou filtro exato (caso tipico do parse_query_syntax)."""
    filters, semantic = parser(query)
    mask = np.array([matches(c, filters) for c in cards])
    if filters.get("cost_pref"):                    # "cheap"/"expensive" so faz sentido para cartas com custo
        mask &= np.array([has_cost(c) for c in cards])

    if verbose:
        active = {key: v for key, v in filters.items() if v not in ([], None, False)}
        print(f"filtros: {active or 'nenhum'} | texto semantico: {semantic or '(nenhum)'!r} | "
              f"{int(mask.sum())}/{len(cards)} cartas passam")
    if not mask.any():
        return []

    # Busca SO por nome (sem texto semantico, sem preferencia de custo): nao precisa de
    # embedding -- mais rapido e nao gasta chamada de modelo/API a toa.
    if filters.get("name") and not semantic and not filters.get("cost_pref"):
        pool = np.flatnonzero(mask)
        order = sorted(pool, key=lambda i: _name_rank(cards[i], filters["name"]))[:k]
        return [{"score": None, "sim": None, "cmc": card_cmc(cards[i]),
                 "name": cards[i]["name"], "type_line": cards[i].get("type_line"),
                 "mana_cost": cards[i].get("mana_cost"), "oracle_text": cards[i].get("oracle_text")}
                for i in order]

    # Consulta SO de filtro, sem nenhum texto semantico pra ranquear (ex.: "t:dinosaur c:g"
    # no parse_query_syntax): nao ha relevancia nenhuma a calcular, entao nem chama o modelo
    # de embedding -- devolve tudo que passou no filtro, em ordem alfabetica pelo nome.
    if alpha_if_no_semantic and not semantic and not filters.get("cost_pref"):
        pool = np.flatnonzero(mask)
        order = sorted(pool, key=lambda i: (cards[i].get("name") or "").lower())[:k]
        return [{"score": None, "sim": None, "cmc": card_cmc(cards[i]),
                 "name": cards[i]["name"], "type_line": cards[i].get("type_line"),
                 "mana_cost": cards[i].get("mana_cost"), "oracle_text": cards[i].get("oracle_text")}
                for i in order]

    text = query if (embed_full_query or not semantic) else semantic
    q = model.encode([query_prefix + text], normalize_embeddings=True)[0]
    sims = embeddings @ q
    scores = np.where(mask, sims, -np.inf)          # quem nao passa no filtro nunca aparece

    pref = filters.get("cost_pref")
    if pref and not semantic:                       # so filtros: ordena por custo
        pool = np.flatnonzero(mask)
        cmcs = np.array([card_cmc(cards[i]) for i in pool], dtype=float)
        sign = 1 if pref == "cheap" else -1
        order = np.lexsort((-sims[pool], sign * cmcs))   # 1o custo, 2o similaridade
        top = [int(pool[j]) for j in order[:k]]
        shown = {i: float(sims[i]) for i in top}
    elif pref:
        pool = [i for i in np.argsort(-scores)[:COST_POOL] if np.isfinite(scores[i])]
        s_pool = np.array([sims[i] for i in pool])
        c_pool = np.array([card_cmc(cards[i]) for i in pool], dtype=float)
        sign = 1 if pref == "cheap" else -1         # cheap: custo alto penaliza; expensive: custo alto premia
        final = _z(s_pool) - sign * COST_WEIGHT * _z(c_pool)
        order = np.argsort(-final)[:k]
        top = [pool[j] for j in order]
        shown = {pool[j]: float(final[j]) for j in order}
    else:
        top = [i for i in np.argsort(-scores)[:k] if np.isfinite(scores[i])]
        shown = {i: float(scores[i]) for i in top}

    return [{"score": shown[i], "sim": float(sims[i]), "cmc": card_cmc(cards[i]),
             "name": cards[i]["name"], "type_line": cards[i].get("type_line"),
             "mana_cost": cards[i].get("mana_cost"), "oracle_text": cards[i].get("oracle_text")}
            for i in top]
