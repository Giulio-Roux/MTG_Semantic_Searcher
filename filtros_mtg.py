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

# Comparacao entre dois stats DA MESMA CARTA (ex.: "toughness greater than power").
# So reconhece campo-contra-campo; um numero do outro lado continua sendo _STAT_PATTERN.
_STAT_COMPARE_PATTERN = re.compile(
    r"\b(power|toughness|loyalty|defense)\s+(?:is\s+)?"
    r"(greater than or equal to|greater than|higher than|more than|exceeds|"
    r"less than or equal to|less than|lower than|"
    r"equal to|the same as|equals)\s+"
    r"(?:its |their |)?(power|toughness|loyalty|defense)\b", re.IGNORECASE)

_STAT_COMPARE_OPS = {
    "greater than or equal to": ">=",
    "greater than": ">", "higher than": ">", "more than": ">", "exceeds": ">",
    "less than or equal to": "<=",
    "less than": "<", "lower than": "<",
    "equal to": "=", "the same as": "=", "equals": "=",
}
ALLOWED_STAT_FIELDS = {"power", "toughness", "loyalty", "defense"}
ALLOWED_STAT_OPS = {">", ">=", "<", "<=", "="}

_PRODUCES_PATTERN = re.compile(
    r"\bproduces?\s+((?:(?:white|blue|black|red|green)\s*(?:,|and|or)?\s*)+)mana\b", re.IGNORECASE)

# Indice dinamico de keywords: minuscula -> nome canonico (como aparece em card["keywords"]).
# NAO e' uma lista fixa digitada a mao -- e' construida a partir do corpus de verdade (set_keyword_index),
# entao cobre exatamente as keywords que existem nos seus dados, com a capitalizacao certa.
# Multi-palavra (ex.: "first strike") funciona: guardamos o numero de palavras de cada entrada
# para tentar casar as frases mais longas primeiro ("double strike" antes de so "strike", se existisse).
# _KEYWORD_COUNT guarda a frequencia de cada keyword (usada para montar o vocabulario do prompt da LLM).
_KEYWORD_INDEX = {}
_KEYWORD_COUNT = {}


def set_keyword_index(cards):
    """Chame uma vez, depois de carregar `cards`, para habilitar o filtro por keyword
    ("flying creature", "creature with first strike and trample").
    Sem chamar isso, filtros de keyword simplesmente nao sao reconhecidos (nao quebra nada)."""
    global _KEYWORD_INDEX, _KEYWORD_COUNT
    idx, cnt = {}, {}
    for c in cards:
        for kw in c.get("keywords") or []:
            k = kw.strip()
            idx.setdefault(k.lower(), k)
            cnt[k.lower()] = cnt.get(k.lower(), 0) + 1
    _KEYWORD_INDEX, _KEYWORD_COUNT = idx, cnt
    return idx


# Indice dinamico de TAGS funcionais (ex.: "removal", "ramp"), no mesmo espirito das keywords:
# construido a partir do proprio corpus (campo TAG_FIELD), nao digitado a mao.
_TAG_INDEX = {}
_TAG_COUNT = {}


def set_tag_index(cards):
    """Chame uma vez, depois de carregar `cards`, para habilitar o filtro por tag
    ("removal creature", "green ramp"). Sem chamar isso, o filtro simplesmente nao e' reconhecido."""
    global _TAG_INDEX, _TAG_COUNT
    idx, cnt = {}, {}
    for c in cards:
        for t in c.get(TAG_FIELD) or []:
            k = t.strip()
            idx.setdefault(k.lower(), k)
            cnt[k.lower()] = cnt.get(k.lower(), 0) + 1
    _TAG_INDEX, _TAG_COUNT = idx, cnt
    return idx


# =====================================================================================
# Indice dinamico de SUPERTIPOS/TIPOS/SUBTIPOS (Legendary, Aura, Dinosaur, Equipment,
# Rabbit, Vehicle, Saga, Desert, ...), no mesmo espirito de _KEYWORD_INDEX / _TAG_INDEX acima:
# construido a partir do proprio type_line dos cards, e nao de uma lista digitada a mao.
# Isso resolve o "nem sempre vai estar escrito Dinosaur": em vez de eu ter que adivinhar/listar
# cada subtipo que existe em Magic, o indice e' extraido dos dados reais, entao cobre
# automaticamente qualquer subtipo que exista no seu corpus, com o nome exatamente como o
# Scryfall escreve (so a CHAVE do dicionario e' minuscula, pra comparar sem depender de
# maiuscula/minuscula; o VALOR guarda a grafia original, caso um dia voce queira exibir/usar).
#
# type_line vem tipicamente como "Legendary Enchantment — Aura" ou "Creature — Dinosaur Warrior":
# a extracao so tira o travessao (em-dash "—", e tambem " - " por seguranca, caso seu dataset
# use hifen simples) e pega cada palavra que sobra, dos dois lados do travessao.
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


def set_all_indexes(cards):
    """Atalho: chama set_keyword_index, set_tag_index e set_type_index de uma vez, habilitando
    os filtros por keyword, tag e subtipo/supertipo (aura, dinosaur, equipment...)."""
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
               "keywords": [], "produced_mana": [], "tags": [], "name": None,
               "stat_compare": []}

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

    # 1c-bis) comparacao entre dois stats DA MESMA CARTA: "toughness greater than power".
    # So funciona campo-contra-campo da propria carta -- comparar com o stat de OUTRA carta
    # (ex.: "defense higher than any creature's power on the field") nao e' representavel aqui
    # e continua caindo inteiro em semantic. Roda ANTES de _STAT_PATTERN para nao deixar
    # "power" ou "toughness" soltos serem mal-interpretados como um stat contra numero.
    def _stat_compare(m):
        field_a, phrase, field_b = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
        op = _STAT_COMPARE_OPS[phrase]
        if field_a != field_b:
            filters["stat_compare"].append((field_a, op, field_b))
        return " "
    rest = _STAT_COMPARE_PATTERN.sub(_stat_compare, rest).strip()

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
        # Subtipos/supertipos dinamicos (aura, dinosaur, equipment, rabbit, saga, ...).
        # Fica DEPOIS do "elif w in TYPE_WORDS" de proposito: os tipos principais continuam
        # sendo resolvidos pelo dicionario fixo (com plural ja mapeado); isso so entra em
        # jogo pra palavras que TYPE_WORDS nao conhece. Guardado em minuscula, porque
        # `matches()` compara com o type_line ja' com .lower() aplicado.
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
    # Comparacao entre dois stats DA MESMA CARTA (ex.: toughness > power). Mesma politica dos
    # outros stats: se algum dos dois lados nao da' pra confirmar como numero ("*", ausente...),
    # exclui em vez de arriscar um match errado.
    for field_a, op, field_b in f.get("stat_compare", []):
        va, vb = _parse_stat(card.get(field_a)), _parse_stat(card.get(field_b))
        if va is None or vb is None:
            return False
        if op == ">" and not (va > vb):
            return False
        if op == ">=" and not (va >= vb):
            return False
        if op == "<" and not (va < vb):
            return False
        if op == "<=" and not (va <= vb):
            return False
        if op == "=" and not (va == vb):
            return False
    return True


# ------------------------------------------------- parser com LLM (IlumA)
ALLOWED_COLORS = set("WUBRG")

LLM_SYSTEM_BASE = """You convert search queries for Magic: The Gathering cards into a JSON filter object.

Return ONLY valid JSON (no markdown fences, no explanations) with exactly these keys:
{"types": [...], "colors": [...], "colorless": false, "cmc_min": null, "cmc_max": null, "cost_pref": null,
 "power_min": null, "power_max": null, "toughness_min": null, "toughness_max": null,
 "loyalty_min": null, "loyalty_max": null, "defense_min": null, "defense_max": null,
 "keywords": [...], "produced_mana": [...], "tags": [...], "name": null,
 "stat_compare": [...], "semantic": "..."}

GOLDEN RULE: nothing in the query may be lost. Every part of the query must end up either in a
structured filter or, when no filter can express it, in "semantic". Never drop a phrase silently.
When a filter captures a phrase, do not repeat that phrase in "semantic". When you are unsure
whether a filter really captures a phrase, put the phrase in "semantic". "semantic" may be empty
only when every part of the query was captured by a filter.

FIELDS

- "types": card types, supertypes and subtypes that the CARD ITSELF has and that the query names.
  Main types: creature, instant, sorcery, artifact, enchantment, land, planeswalker, battle.
  Supertypes: legendary, basic, snow, world. Subtypes: any creature type (dinosaur, angel, elf...),
  artifact subtype (equipment, vehicle, food, clue...), enchantment subtype (aura, saga, class...),
  land subtype (desert, cave, gate...), etc. Write every entry lowercase and singular.
  A type or subtype that is only the TARGET of the card's effect does NOT go in "types":
  "dinosaur that costs 7 mana" -> types ["dinosaur"] (the card is a dinosaur);
  "destroy all dinosaurs" -> types [] and the phrase stays in "semantic" (dinosaurs are the target).
  Include a type only if that word (or its plain singular) literally appears in the query. Never infer
  a main type from a tag or effect ("removal" does not imply instant/sorcery), and never add "creature"
  just because a creature subtype was named. If unsure that a word is a real Magic type, leave it in "semantic".
  In English noun compounds the LAST noun is the head: "creature destroyer artifact" is an ARTIFACT that
  destroys creatures.
- "colors": colors the card must have, letters W U B R G ("blue-black" -> ["U","B"], the card must have both).
  "colorless": true only if the query asks for colorless cards.
- "cost_pref": "cheap" or "expensive" if the query asks for it, else null. Never turn it into numbers.
- Numeric limits ("cmc_*", "power_*", "toughness_*", "loyalty_*", "defense_*") are INCLUSIVE integers:
  "N or greater"/"at least N" -> min=N; "N or less"/"at most N" -> max=N;
  "more than N" -> min=N+1; "less than N" -> max=N-1; "exactly N"/"costs N" -> min=max=N.
  Only EXPLICIT numbers go here ("costs zero" -> 0). Use power/toughness for creatures and vehicles,
  loyalty for planeswalkers, defense for battles.
- "stat_compare": list of [field_a, op, field_b] comparing two numeric stats OF THE SAME CARD
  (fields: "power", "toughness", "loyalty", "defense"; op one of ">", ">=", "<", "<=", "=").
  "toughness higher than power" -> [["toughness", ">", "power"]]. Use it ONLY when both sides are stats of
  the card itself. A comparison against a fixed number belongs in the *_min/*_max fields; a comparison against
  anything else (other cards, the board, life totals, counts of permanents) stays in "semantic".
- "keywords": abilities or actions the CARD ITSELF has, chosen ONLY from the VALID KEYWORDS list at the end
  (copy that exact spelling). Match by meaning, not by spelling: the query may use another grammatical form
  ("discovers" -> "Discover", "proliferates" -> "Proliferate", "flies" -> "Flying", "with first strike" ->
  "First strike"). A keyword the effect GIVES to other permanents ("creatures you control gain flying") is NOT a
  keyword of the card: keep that phrase in "semantic". If nothing in the list matches, keep the phrase in "semantic".
- "tags": functional categories, chosen ONLY from the VALID TAGS list at the end. Use a tag only when the query asks
  for that CATEGORY of card ("removal", "ramp", "card advantage"), not when it describes one specific effect.
- "produced_mana": colors a land or mana rock PRODUCES; the card must produce ALL colors listed.
  Requests such as "two or more colors" or "any color" cannot be expressed here: keep them in "semantic".
- "name": set ONLY when the query asks for one SPECIFIC named card (quoted text, "the card Sol Ring", or just a card
  name). Do not set it when the query describes what a card does.
- "semantic": English phrase describing what the card DOES, without the words already used by filters."""


def _vocab_line(index, count, limit):
    keys = sorted(index, key=lambda k: (-count.get(k, 0), k))[:limit]
    return ", ".join(sorted(index[k] for k in keys))


def build_llm_system():
    """Prompt base + vocabulario REAL do corpus (keywords/tags), lido dos indices na hora da chamada."""
    kw = _vocab_line(_KEYWORD_INDEX, _KEYWORD_COUNT, 400) or "(none loaded)"
    tg = _vocab_line(_TAG_INDEX, _TAG_COUNT, 120) or "(none loaded)"
    return (LLM_SYSTEM_BASE
            + "\n\nVALID KEYWORDS: " + kw
            + "\n\nVALID TAGS: " + tg)


def _ex(types=(), colors=(), cmc_min=None, cmc_max=None, cost_pref=None,
       power_min=None, power_max=None, toughness_min=None, toughness_max=None,
       loyalty_min=None, loyalty_max=None, defense_min=None, defense_max=None,
       keywords=(), produced_mana=(), tags=(), name=None, stat_compare=(),
       colorless=False, semantic=""):
    return {"types": list(types), "colors": list(colors), "colorless": colorless,
            "cmc_min": cmc_min, "cmc_max": cmc_max, "cost_pref": cost_pref,
            "power_min": power_min, "power_max": power_max,
            "toughness_min": toughness_min, "toughness_max": toughness_max,
            "loyalty_min": loyalty_min, "loyalty_max": loyalty_max,
            "defense_min": defense_min, "defense_max": defense_max,
            "keywords": list(keywords), "produced_mana": list(produced_mana), "tags": list(tags),
            "name": name, "stat_compare": [list(x) for x in stat_compare], "semantic": semantic}


_LLM_EXAMPLES = [
    # --- tipos/subtipos: a propria carta x alvo do efeito ---
    ("creature that destroys enchantment", _ex(["creature"], semantic="destroys enchantment")),
    ("artifact destroyer artifact", _ex(["artifact"], semantic="destroys artifacts")),
    ("destroy target creature", _ex(semantic="destroy target creature")),
    ("destroy all dinosaurs", _ex(semantic="destroy all dinosaurs")),
    ("dinosaur that costs 7 mana", _ex(["dinosaur"], cmc_min=7, cmc_max=7)),
    ("equipment that gives +2/+2", _ex(["equipment"], semantic="gives +2/+2")),
    ("saga that draws cards", _ex(["saga"], semantic="draws cards")),
    ("indestructible green enchantment", _ex(["enchantment"], ["G"], keywords=["Indestructible"])),

    # --- cores ---
    ("blue-black instant that counters spells", _ex(["instant"], ["U", "B"], semantic="counters spells")),
    ("colorless artifact", _ex(["artifact"], colorless=True)),

    # --- custo ---
    ("instant that counters a spell and costs three mana or less",
     _ex(["instant"], cmc_max=3, semantic="counters a spell")),
    ("artifact that costs more than 4 mana", _ex(["artifact"], cmc_min=5)),

    # --- stats numericos contra constante ---
    ("vehicle with power 4 or greater", _ex(["vehicle"], power_min=4)),
    ("vehicle with power 1 or less", _ex(["vehicle"], power_max=1)),
    ("creature with toughness 5 or greater", _ex(["creature"], toughness_min=5)),
    ("creature with toughness 1 or less that has flying",
     _ex(["creature"], toughness_max=1, keywords=["Flying"])),
    ("planeswalker with loyalty 6 or higher that draws cards",
     _ex(["planeswalker"], loyalty_min=6, semantic="draws cards")),
    ("planeswalker with loyalty 3 or lower", _ex(["planeswalker"], loyalty_max=3)),
    ("battle with defense 3 or less", _ex(["battle"], defense_max=3)),
    ("battle with defense 5 or greater that deals damage",
     _ex(["battle"], defense_min=5, semantic="deals damage")),

    # --- comparacao entre dois stats DA MESMA CARTA ---
    ("creature with toughness greater than power",
     _ex(["creature"], stat_compare=[("toughness", ">", "power")])),
    ("vehicle whose power is equal to its toughness",
     _ex(["vehicle"], stat_compare=[("power", "=", "toughness")])),
    ("angel that scries, with toughness higher than power",
     _ex(["angel"], keywords=["Scry"], stat_compare=[("toughness", ">", "power")])),
    # comparacao com algo que NAO e' stat da propria carta: nao ha filtro -> semantic
    ("battle with defense higher than any creature's power on the field",
     _ex(["battle"], semantic="defense higher than any creature's power on the field")),

    # --- keywords (inclusive formas verbais; keyword DADA a outros vai pra semantic) ---
    ("cheap green creature with flying", _ex(["creature"], ["G"], cost_pref="cheap", keywords=["Flying"])),
    ("permanent with flying and trample", _ex(keywords=["Flying", "Trample"])),
    ("artifact that proliferates", _ex(["artifact"], keywords=["Proliferate"])),
    ("elf that investigates and draws a card",
     _ex(["elf"], keywords=["Investigate"], semantic="draws a card")),
    ("gives creatures you control flying", _ex(semantic="gives creatures you control flying")),

    # --- produced_mana (a carta precisa produzir TODAS as cores listadas) ---
    ("land that produces green mana", _ex(["land"], produced_mana=["G"])),
    ("land that produces two or more colors of mana",
     _ex(["land"], semantic="produces two or more colors of mana")),

    # --- tags (sem assumir tipo) ---
    ("cheap green removal", _ex([], ["G"], cost_pref="cheap", tags=["removal"])),
    ("removal that generates card advantage", _ex(tags=["removal", "card-advantage"])),
    ("ramp that fixes colors", _ex(tags=["ramp"], semantic="fixes colors")),
    ("card advantage that costs two mana or less", _ex(cmc_max=2, tags=["card-advantage"])),

    # --- nome ---
    ('"Lightning Bolt"', _ex(name="Lightning Bolt")),
    ("the card Sol Ring", _ex(name="Sol Ring")),
]


def _llm_messages(query):
    msgs = [{"role": "system", "content": build_llm_system()}]
    for q, answer in _LLM_EXAMPLES:                      # few-shot: pares user/assistant escritos por nos
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": json.dumps(answer)})
    msgs.append({"role": "user", "content": query})
    return msgs


def _strip_fences(text: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()  # modelos com raciocinio
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:                      # texto antes/depois do JSON
        t = t[i:j + 1]
    return t


def _as_int(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str) and x.strip().lstrip("-").isdigit():   # "4" -> 4
        return int(x.strip())
    return None


def _valid_stat_compare(entry):
    """Aceita so [field_a, op, field_b] bem formado, com os dois campos entre os quatro
    stats conhecidos, op entre os cinco operadores permitidos, e field_a != field_b
    (comparar um stat com ele mesmo nao tem sentido)."""
    if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
        return None
    a, op, b = entry
    if not (isinstance(a, str) and isinstance(op, str) and isinstance(b, str)):
        return None
    a, b = a.lower(), b.lower()
    if a in ALLOWED_STAT_FIELDS and b in ALLOWED_STAT_FIELDS and op in ALLOWED_STAT_OPS and a != b:
        return (a, op, b)
    return None


def _lookup_inflected(word, index):
    """Procura `word` no indice tolerando plural/flexao verbal
    ("discovers" -> "discover", "scries" -> "scry", "proliferating" -> "proliferate")."""
    w = word.strip().lower()
    if not w:
        return None
    cands = [w]
    if w.endswith("ies"):
        cands.append(w[:-3] + "y")
    if w.endswith("es"):
        cands.append(w[:-2])
    if w.endswith("s"):
        cands.append(w[:-1])
    if w.endswith("ed"):
        cands += [w[:-2], w[:-1]]
    if w.endswith("ing"):
        cands += [w[:-3], w[:-3] + "e"]
    for c in cands:
        if c in index:
            return index[c]
    return None


_NUM_KEYS = ("cmc_min", "cmc_max", "power_min", "power_max", "toughness_min", "toughness_max",
             "loyalty_min", "loyalty_max", "defense_min", "defense_max")


def _validate(data: dict, report=None):
    """Aceita so valores validos, mas NUNCA perde informacao em silencio:
    - tudo que for descartado e' anotado em `report` (lista de strings, p/ debug);
    - tipo/keyword/tag que nao validou volta para o texto semantico, entao o embedding ainda o ve."""
    if report is None:
        report = []
    leftovers = []

    def _list(key):
        v = data.get(key)
        return v if isinstance(v, list) else []

    types = []
    for t in _list("types"):
        if not isinstance(t, str):
            continue
        tl = t.strip().lower()
        canon = TYPE_WORDS.get(tl) or (_type_index_lookup(tl) or "").lower() or None
        if canon:
            types.append(canon)
        else:
            report.append(f"types: {t!r} nao existe no corpus (indice de tipos tem {len(_TYPE_INDEX)} entradas)")
            leftovers.append(tl)

    colors = []
    for c in _list("colors"):
        if isinstance(c, str) and c.upper() in ALLOWED_COLORS:
            colors.append(c.upper())
        else:
            report.append(f"colors: {c!r} invalida")

    produced = []
    for c in _list("produced_mana"):
        if isinstance(c, str) and c.upper() in ALLOWED_COLORS:
            produced.append(c.upper())
        else:
            report.append(f"produced_mana: {c!r} invalida")

    keywords = []
    for k in _list("keywords"):
        canon = _lookup_inflected(k, _KEYWORD_INDEX) if isinstance(k, str) else None
        if canon:
            keywords.append(canon)
        else:
            report.append(f"keywords: {k!r} nao esta no indice (indice tem {len(_KEYWORD_INDEX)} entradas)")
            if isinstance(k, str):
                leftovers.append(k)

    tags = []
    for t in _list("tags"):
        canon = _lookup_inflected(t, _TAG_INDEX) if isinstance(t, str) else None
        if canon:
            tags.append(canon)
        else:
            report.append(f"tags: {t!r} nao esta no indice (indice tem {len(_TAG_INDEX)} entradas)")
            if isinstance(t, str):
                leftovers.append(t)

    stat_compare = []
    for e in _list("stat_compare"):
        sc = _valid_stat_compare(e)
        if sc:
            stat_compare.append(sc)
        else:
            report.append(f"stat_compare: entrada invalida {e!r}")

    nums = {}
    for k in _NUM_KEYS:
        raw = data.get(k)
        nums[k] = _as_int(raw)
        if raw is not None and nums[k] is None:
            report.append(f"{k}: valor nao numerico {raw!r}")

    name = data.get("name")
    name = name.strip() if isinstance(name, str) and name.strip() else None
    semantic = data.get("semantic")
    semantic = semantic.strip() if isinstance(semantic, str) else ""
    semantic = " ".join([semantic] + leftovers).strip()

    cost_pref = data.get("cost_pref")
    filters = {"types": list(dict.fromkeys(types)), "colors": list(dict.fromkeys(colors)),
               "colorless": data.get("colorless") is True,
               "cost_pref": cost_pref if cost_pref in ("cheap", "expensive") else None,
               "keywords": list(dict.fromkeys(keywords)), "produced_mana": list(dict.fromkeys(produced)),
               "tags": list(dict.fromkeys(tags)), "name": name,
               "stat_compare": list(dict.fromkeys(stat_compare)), **nums}
    return filters, semantic


_llm_cache = {}


def parse_query_llm(query, client, model="iluma", temperature=0.5, fallback=parse_query,
                    debug=False, use_cache=True):
    """Interpreta a consulta com o LLM. Se falhar (timeout, JSON ruim...), usa o parser de regras.
    temperature=0.5: abaixo disso a IlumA trava (ver material da aula).
    debug=True: mostra o JSON BRUTO da IlumA e tudo que o _validate descartou/moveu pro semantic.
    use_cache=False: ignora _llm_cache (util ao testar mudancas no prompt)."""
    key = query.strip().lower()
    if use_cache and key in _llm_cache:                  # mesma consulta = nao gasta cota de novo
        if debug:
            print("(resultado vindo do cache; use use_cache=False ou _llm_cache.clear())")
        return _llm_cache[key]
    try:
        r = client.chat.completions.create(model=model, messages=_llm_messages(query),
                                           temperature=temperature)
        raw = r.choices[0].message.content
        report = []
        result = _validate(json.loads(_strip_fences(raw)), report)
        if debug:
            print("LLM bruto:", raw.strip())
            for line in report:
                print("  descartado/ajustado ->", line)
            if not report:
                print("  (nada descartado no _validate)")
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
#   cmp:campoAopcampoB   comparacao entre dois stats DA MESMA CARTA (ex.: "cmp:tou>pow" ==
#               toughness maior que power). Operadores: > >= < <= =. Campos aceitos: power,
#               toughness, loyalty, defense (ou as abreviacoes pow/tou/loy/def).
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
                "cmp", "k", "pm", "tag", "cost")
_SYNTAX_TOKEN = re.compile(
    r"(?<!\S)(" + "|".join(_SYNTAX_KEYS) + r")(:|<=|>=|=|<|>)(?:\(([^)]*)\)|(\S+))",
    re.IGNORECASE)
_SYNTAX_STAT_FIELD = {"cmc": "cmc", "mv": "cmc", "pow": "power", "tou": "toughness",
                       "loy": "loyalty", "def": "defense"}
_STAT_ABBR = {"pow": "power", "tou": "toughness", "loy": "loyalty", "def": "defense",
              "power": "power", "toughness": "toughness", "loyalty": "loyalty", "defense": "defense"}
_CMP_VALUE = re.compile(
    r"\s*(power|toughness|loyalty|defense|pow|tou|loy|def)\s*(>=|<=|>|<|=)\s*"
    r"(power|toughness|loyalty|defense|pow|tou|loy|def)\s*$", re.IGNORECASE)


def _syntax_words(raw):
    """Divide um valor (bruto, de dentro ou fora de parenteses) em palavras minusculas,
    sem pontuacao de borda -- usado por o:/t:/k:/tag: quando o valor vem entre parenteses
    (ex.: "o:(deals damage)" filtra as duas palavras, ambas obrigatorias). Mantem hifen
    dentro da palavra (varias tags/keywords usam hifen, ex. "card-advantage")."""
    return [w for w in re.sub(r"[^\w\s-]", " ", raw.lower()).split() if w]


def parse_query_syntax(query: str):
    """Parser por sintaxe explicita (`o:`, `t:`, `n:(...)`, `c:`, `cmc:`, `pow:`, `cmp:`, `k:`,
    `pm:`, `tag:`, `cost:` -- ver comentario acima da lista completa). Devolve (filtros, texto
    semantico), no mesmo formato de parse_query/parse_query_llm, entao funciona direto como
    `parser=parse_query_syntax` no hybrid_search."""
    filters = {"types": [], "colors": [], "colorless": False,
               "cmc_min": None, "cmc_max": None, "cost_pref": None,
               "power_min": None, "power_max": None, "toughness_min": None, "toughness_max": None,
               "loyalty_min": None, "loyalty_max": None, "defense_min": None, "defense_max": None,
               "keywords": [], "produced_mana": [], "tags": [], "name": None, "oracle_words": [],
               "stat_compare": []}

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
        elif key == "cmp":
            # sintaxe: cmp:tou>pow  (o operador fica DENTRO do valor, depois do ':')
            mc = _CMP_VALUE.match(valor)
            if mc:
                field_a = _STAT_ABBR[mc.group(1).lower()]
                cmp_op = mc.group(2)
                field_b = _STAT_ABBR[mc.group(3).lower()]
                if field_a != field_b:
                    filters["stat_compare"].append((field_a, cmp_op, field_b))
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


def _z(x):
    """Padroniza (media 0, desvio 1) para poder somar relevancia e custo na mesma escala."""
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def hybrid_search(query, model, embeddings, cards, query_prefix, k=10, verbose=True, parser=parse_query,
                  embed_full_query=False, alpha_if_no_semantic=False):
    """cards: lista de dicts ALINHADA com as linhas de `embeddings` (mesma ordem).
    Com "cheap"/"expensive" na consulta, o campo 'score' vira uma nota combinada
    (relevancia + custo); a similaridade pura fica em 'sim'.
    Consulta so de filtros + cheap/expensive (ex.: "cheap green creature"): nao ha relevancia
    a preservar, entao ordena direto por custo (desempate pela similaridade).
    embed_full_query=False (padrao): embeda so o texto que sobrou depois de tirar os filtros
    (ex.: 'destroys artifacts'; se nao sobrou nada, embeda a consulta inteira);
    True: embeda a consulta INTEIRA, mesmo depois de tirar os filtros.
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