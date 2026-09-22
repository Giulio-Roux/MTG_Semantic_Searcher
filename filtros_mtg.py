"""Filtros por metadados + busca hibrida para o MTG Semantic Searcher.

Ideia: separar a consulta em duas partes.
  1. FILTROS (tipo, cor, custo)  -> codigo comum, garantido, sem ambiguidade.
  2. SIGNIFICADO (o resto)       -> embedding, que ordena as cartas que passaram.

Regra para nao confundir assunto com alvo:
  so as palavras da FRASE NOMINAL INICIAL viram filtro.
  "creature that destroys artifacts"  -> filtro: Creature   | semantico: "destroys artifacts"
  "destroy target creature"           -> sem filtro         | semantico: a frase inteira
"""
import json
import re
import numpy as np

# ---------------------------------------------------------------- parametros
# "cheap"/"expensive" NAO viram corte (cmc <= 2): viram uma PREFERENCIA na ordenacao.
# Entre as COST_POOL cartas mais relevantes, o custo entra como um desempate suave.
COST_POOL = 100      # quantas cartas (as mais relevantes) sao reordenadas por custo
COST_WEIGHT = 0.5    # peso do custo: 0 = ignora o custo; 1 = custo pesa tanto quanto a relevancia

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
FILLERS = {"a", "an", "the", "some", "mana"}   # palavras que podem aparecer na frase inicial sem mudar nada

_CMC_PATTERN = re.compile(r"\b(?:cmc|mana value)\s*(<=|>=|=|<|>)?\s*(\d+)\b", re.IGNORECASE)


# ---------------------------------------------------------- interpretar a consulta
def parse_query(query: str):
    """Devolve (filtros, texto_semantico)."""
    filters = {"types": [], "colors": [], "colorless": False,
               "cmc_min": None, "cmc_max": None, "cost_pref": None}

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


def matches(card: dict, f: dict) -> bool:
    type_line = (card.get("type_line") or "").lower()
    if any(t not in type_line for t in f["types"]):
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
    return True


# ------------------------------------------------- parser com LLM (IlumA)
ALLOWED_TYPES = set(TYPE_WORDS.values())
ALLOWED_COLORS = set("WUBRG")

LLM_SYSTEM = """You convert search queries for Magic: The Gathering cards into JSON.

Return ONLY valid JSON, no markdown fences, with exactly these keys:
{"types": [...], "colors": [...], "colorless": false, "cmc_min": null, "cmc_max": null, "cost_pref": null, "semantic": "..."}

Rules:
- "types": card types the CARD ITSELF must have (creature, instant, sorcery, artifact,
  enchantment, land, planeswalker, battle, legendary). A type that is only the TARGET of the
  card's effect does NOT go in "types".
- In English noun compounds the LAST noun is the head: "creature destroyer artifact" is an
  ARTIFACT that destroys creatures; "artifact destroyer creature" is a CREATURE that destroys artifacts.
- "colors": colors the card must have, as letters W U B R G. "colorless": true only if asked.
- "cost_pref": "cheap" or "expensive" if the query asks for it, else null. Do NOT turn it into numbers.
- Only EXPLICIT costs ("cmc <= 3", "mana value 2") go into cmc_min/cmc_max.
- "semantic": English phrase describing what the card DOES, without the words used for filters.
  If the query is ONLY filters (no effect described), use an empty string."""

def _ex(types=(), colors=(), cmc_min=None, cmc_max=None, cost_pref=None, semantic=""):
    return {"types": list(types), "colors": list(colors), "colorless": False,
            "cmc_min": cmc_min, "cmc_max": cmc_max, "cost_pref": cost_pref, "semantic": semantic}


_LLM_EXAMPLES = [
    ("creature that destroys artifacts", _ex(["creature"], semantic="destroys artifacts")),
    ("creature destroyer artifact", _ex(["artifact"], semantic="destroys creatures")),
    ("cheap green creature with flying", _ex(["creature"], ["G"], cost_pref="cheap", semantic="flying")),
    ("destroy target creature", _ex(semantic="destroy target creature")),
    ("cheap green creature", _ex(["creature"], ["G"], cost_pref="cheap", semantic="")),
    ("instant that counters a spell, cmc <= 3", _ex(["instant"], cmc_max=3, semantic="counters a spell")),
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
    """Aceita so valores permitidos: o que o LLM inventar fora das listas e descartado."""
    types = [t.lower() for t in data.get("types", []) if isinstance(t, str) and t.lower() in ALLOWED_TYPES]
    colors = [c.upper() for c in data.get("colors", []) if isinstance(c, str) and c.upper() in ALLOWED_COLORS]
    semantic = data.get("semantic")
    if not isinstance(semantic, str):
        semantic = ""
    filters = {"types": list(dict.fromkeys(types)), "colors": list(dict.fromkeys(colors)),
               "colorless": data.get("colorless") is True,
               "cmc_min": _as_int(data.get("cmc_min")), "cmc_max": _as_int(data.get("cmc_max")),
               "cost_pref": data.get("cost_pref") if data.get("cost_pref") in ("cheap", "expensive") else None}
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
        print(f"(LLM falhou: {type(e).__name__}; usando o parser de regras)")
        return fallback(query)
    _llm_cache[key] = result
    return result


# ------------------------------------------------------------- busca hibrida
def _z(x):
    """Padroniza (media 0, desvio 1) para poder somar relevancia e custo na mesma escala."""
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def hybrid_search(query, model, embeddings, cards, query_prefix, k=10, verbose=True, parser=parse_query,
                  embed_full_query=True):
    """cards: lista de dicts ALINHADA com as linhas de `embeddings` (mesma ordem).
    Com "cheap"/"expensive" na consulta, o campo 'score' vira uma nota combinada
    (relevancia + custo); a similaridade pura fica em 'sim'.
    Consulta so de filtros + cheap/expensive (ex.: "cheap green creature"): nao ha relevancia
    a preservar, entao ordena direto por custo (desempate pela similaridade).
    embed_full_query=True (padrao): embeda a consulta INTEIRA, mesmo depois de tirar os filtros;
    False: embeda so o texto que sobrou (ex.: 'destroys artifacts')."""
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