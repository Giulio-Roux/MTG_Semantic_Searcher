<img src="./Cabecalho.jpg"/>

# MTG Semantic Searcher

Buscador semântico em linguagem natural para cartas de *Magic: The Gathering*, desenvolvido como projeto da disciplina de Processamento de Linguagem Natural da [Ilum – Escola de Ciência (CNPEM)](https://ilum.cnpem.br/).

> Em vez de aprender a sintaxe do Scryfall, escreva o que você quer:
> `green creatures that cost two mana and have flying`, `cards that go wide`, `sacrifice permanents I control to generate value`.

---

## Motivação

*Magic: The Gathering* tem mais de 35 mil cartas e mais de 20 formatos (agosto de 2026). Montar um baralho exige avaliar centenas ou milhares de cartas e suas interações, o que é especialmente difícil para jogadores novos.

Buscadores como o [Scryfall](https://scryfall.com) e o [Gatherer](https://gatherer.wizards.com) são precisos, mas têm duas limitações:

- **Inteligibilidade:** é preciso aprender uma linguagem de consulta própria antes de fazer buscas úteis.
- **Criatividade:** a busca é literal, e o jogo tem muitos efeitos diferentes com o mesmo resultado prático (por exemplo, "sacrifice" é uma mecânica que aparece de diferentes formas: há múltiplas maneiras de sacrificar as suas permanentes em troca de algo).

Este projeto combina **similaridade semântica** (embeddings + similaridade de cossenos) com **filtros estruturados** (cor, tipo, custo de mana etc.) para tornar a busca mais acessível e menos presa à literalidade do texto.

## Como funciona

```
Scryfall (oracle-cards) + Comprehensive Rules
        │
        ▼
 dicionário de keywords ──► documentos de carta ──► embeddings
   (Etapa 1)                  (Etapa 1)              (Etapa 2)
                                                        │
 pedido do usuário ──► filtros + texto ─────────────────┤
                                                        ▼
                         filtragem por metadados + similaridade de cossenos
                                                        │  (Etapa 3)
                                                        ▼
                                                 top-k cartas
```

O pedido é dividido em duas partes: **filtros** aplicados aos metadados das cartas e **texto semântico** que é embedado e comparado por cosseno com os vetores das cartas que passaram nos filtros. Por padrão (`embed_full_query=False` em `hybrid_search`), só o texto que sobrou depois de retirar os filtros é embedado; se não sobrar nada, a consulta inteira é usada. Com `embed_full_query=True`, a consulta inteira é sempre embedada.

Quem faz a divisão em filtros e texto depende da função de busca usada em `busca_mtg.ipynb`:

| Função | Como o pedido é interpretado |
|---|---|
| `search(query)` | Só embedding, sem filtros (linha de base). |
| `search2(query)` | Busca híbrida: filtros extraídos por regras em Python (`parse_query`). |
| `search3(query)` | **Linguagem natural pura:** uma LLM extrai os filtros (JSON). Se a API falhar ou der *timeout*, cai para o `parse_query` em Python. |
| `search4(query)` | **Sintaxe explícita** (`t:creature c:g cmc:2 k:flying`) misturada com texto livre. Sem LLM. |

`search3` é indicada para quem não conhece a sintaxe, porque a LLM encontra os filtros no próprio texto. `search4` é mais rápida e dá controle total a jogadores experientes.

### Filtros suportados

Tipos e subtipos (extraídos dinamicamente do corpus), cores, cor "colorless", custo de mana, poder, resistência, lealdade, defesa (com mínimo e máximo), comparação entre dois atributos da mesma carta (por exemplo, resistência maior que poder), keywords, cores de mana produzidas, tags funcionais (`removal`, `ramp`, `card-advantage`), nome da carta e preferência de custo (`cheap` / `expensive`, que reordena os resultados em vez de cortar).

Na `search3`, o prompt da LLM recebe o vocabulário real de keywords e tags do corpus, e o que a LLM não consegue expressar como filtro deve voltar para o texto semântico, para não se perder. Valores inventados que não existem no corpus são descartados na validação (`parse_query_llm(..., debug=True)` mostra o JSON bruto e o que foi descartado).

Sintaxe de `search4`:

| Token | Filtra por | Exemplo |
|---|---|---|
| `t:` | tipo, subtipo, supertipo | `t:dinosaur` |
| `c:` | cores (letras `wubrg`) ou `colorless` | `c:rg` |
| `cmc:` / `mv:` | custo de mana (`:`, `=`, `<`, `<=`, `>`, `>=`) | `cmc<=3` |
| `pow:` `tou:` `loy:` `def:` | poder, resistência, lealdade, defesa | `pow>=4` |
| `cmp:` | compara dois atributos da mesma carta (`>`, `>=`, `<`, `<=`, `=`) | `cmp:tou>pow` |
| `k:` | keyword (repita para AND) | `k:flying k:trample` |
| `pm:` | cores de mana que a carta produz | `pm:g` |
| `tag:` | tag funcional (repita para AND) | `tag:removal` |
| `o:` | palavras obrigatórias no texto de regras | `o:(deals damage)` |
| `n:` | nome da carta | `n:"Lightning Bolt"` |
| `cost:` | `cheap` ou `expensive` (preferência, não corte) | `cost:cheap` |

Todo o resto da consulta vai para o embedding. Se a consulta for só de filtros, `search4` pula o embedding e devolve as cartas em ordem alfabética.

## Dados e pré-processamento

- **Cartas:** `oracle-cards.jsonl`, bulk data da [API do Scryfall](https://scryfall.com/docs/api/bulk-data).
- **Regras:** *Comprehensive Rules* em `.txt`, da Wizards of the Coast.
- Os arquivos usados nos experimentos foram baixados em **17/08/2026**.

Principais decisões:

1. **Corpus:** foram removidos *tokens*, emblemas, cartas de arte, cartas de formatos alternativos (Vanguard, Archenemy, Planechase) e *Un-sets*. O corpus final tem **33.147 cartas**.
2. **Dicionário de keywords:** montado a partir das seções 701 (*Keyword Actions*) e 702 (*Keyword Abilities*) das regras, com o glossário como reserva. Cada entrada guarda `definition`, `raw`, `source` e `category`.
3. **Texto de embedding** de cada carta, em até três blocos: `Oracle Text`, `Keyword Definitions` (definição de cada keyword da carta, com o parâmetro quando existe, como `Cycling {2}`) e `Functions` (otags). Nome, tipo, custo e cores **não** entram no texto, só nos metadados, para que cartas com o mesmo efeito gerem o mesmo texto. Cartas sem texto de efeito recebem o texto `vanilla` (e, em cartas de duas faces, cada face sem texto), então nenhuma carta fica de fora dos embeddings.
4. **Otags do Scryfall:** `ramp`, `card-advantage` e `removal`, obtidas pela API de busca (`otag:<tag>`) e guardadas em cache.
5. **Normalização do texto:** símbolos de mana são escritos por extenso e habilidades ativadas `custo: efeito` viram "faça custo para fazer efeito" (`{T}: Add {G}{G}` → *Tap this permanent to add two green mana*).
6. **Embedding:** [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (1024 dimensões, `max_seq_length` 1024). A consulta recebe um prefixo de instrução e os documentos não. O modelo usa `padding_side="left"`, tanto na geração dos vetores quanto na busca. 

## Estrutura do repositório

| Arquivo | Descrição |
|---|---|
| `dicionario_keywords_mtg.ipynb` | **Etapa 1.** Gera `keyword_dictionary.json` a partir das *Comprehensive Rules*. |
| `documentos_cartas_mtg.ipynb` | **Etapa 1.** Filtra o corpus, busca as otags, monta o texto de embedding e salva `card_documents.jsonl`. |
| `embeddings_busca_mtg_qwen3.ipynb` | **Etapa 2.** Gera os embeddings com Qwen3, em blocos com checkpoint. Preparado para rodar em GPU/HPC, inclusive offline. |
| `busca_mtg.ipynb` | **Etapa 3.** Carrega os vetores e define `search`, `search2`, `search3` e `search4`. |
| `filtros_mtg.py` | Parsers de consulta (regras, LLM e sintaxe), filtros por metadados e `hybrid_search`. |

## Como usar

### 1. Instalação

```bash
git clone https://github.com/Giulio-Roux/MTG_Semantic_Searcher.git
cd MTG_Semantic_Searcher

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy requests openai jupyter torch "sentence-transformers>=2.7.0" "transformers>=4.51.0"
```

`openai` só é necessário para `search3` (a API da LLM é acessada pelo cliente compatível com OpenAI) e `requests` para baixar as otags do Scryfall.

### 2. Dados

Os dados não estão versionados (os arquivos são grandes demais para o git). Baixe-os e organize a pasta `Dados/` assim:

```
Dados/
├── Raw/
│   ├── oracle-cards.jsonl                 # Scryfall (bulk data "Oracle Cards")
│   └── MTG_Comprehensive_Rules.txt        # Comprehensive Rules da WOTC
├── Etapa1_card_expanded/                  # gerada pelos notebooks da Etapa 1
│   ├── keyword_dictionary.json
│   ├── keyword_dictionary_ability_words.json
│   ├── card_otags.json
│   ├── otag_cache/
│   └── card_documents.jsonl
└── Etapa2_embeddings/                     # gerada pelo notebook da Etapa 2
    ├── embeddings_qwen3.npy
    ├── embeddings_qwen3_meta.json
    └── checkpoints_qwen3/
```

Se for versionar o projeto, inclua `Dados/` em um `.gitignore`.

### 3. Ordem de execução

1. `dicionario_keywords_mtg.ipynb`
2. `documentos_cartas_mtg.ipynb`
3. `embeddings_busca_mtg_qwen3.ipynb`
4. `busca_mtg.ipynb`

### 4. Buscando

Em `busca_mtg.ipynb`:

```python
# linguagem natural (usa a LLM para extrair os filtros)
print_results(search3("green creatures that cost two mana and have flying", k=5))

# sintaxe explícita + texto livre (sem LLM)
print_results(search4("t:dinosaur Discover when dinosaur enters", k=10))
```

### 5. API da LLM (`search3`)

O notebook usa um servidor compatível com a API da OpenAI (o da IlumA, na rede do CNPEM). O token é lido da variável de ambiente `ILUMA_TOKEN` ou pedido por `getpass`:

```bash
export ILUMA_TOKEN="sk-..."
```

Para usar outro provedor, troque `base_url` e `model` na célula que cria o cliente `OpenAI(...)` em `busca_mtg.ipynb`. Sem a API, `search3` usa o *fallback* em Python.

## Resultados

Os testes foram qualitativos, feitos com pedidos escritos à mão. Eles foram feitos com uma versão do parser anterior à comparação entre atributos (`stat_compare` / `cmp:`) e não foram refeitos com a versão atual do código.

**O que funcionou bem**

- **Múltiplos filtros em uma só frase:** "*green creatures that cost two mana and have flying*" gerou os 4 filtros corretos (cor, tipo, custo e keyword), reduzindo 33.147 cartas a 28 candidatas.
- **Busca por nome:** buscas retornaram a carta certa, inclusive com palavras de enchimento ("*I'm looking for…*") e com cartas de duas faces. Mas no momento não considera possíveis erros de digitação.
- **Sinônimos simples:** "*remove lands*" retornou cartas que destroem ou exilam terrenos.
- **Gatilhos:** "*destroy a land when they enter*" e "*draw a card when they enter*" trouxeram cada uma 5 cartas corretas no top 5.
- **Estabilidade da API:** mais de 100 buscas em sequência sem *timeout* em condições adequadas.
- **Reescrita semântica pela LLM:** "*go wide*" virou algo como "*creates many creatures/tokens*", o que melhorou o resultado do embedding.
- **Ambiguidade:** a LLM interpreta melhor que o *fallback* em Python frases como "*creature destroyer*" e "*destroy green cards*".

**Limitações conhecidas**

- **Negações e exceções:** "*removes creatures without destroying them*" e "*destroy all creatures except yours*" retornam cartas do efeito oposto.
- **Generalizadores** (`all`, `any`): só funcionam dentro de expressões comuns nas cartas (como "*destroy all creatures*").
- **Posse e referência:** "*opponents creatures*", "*itself*", "*themselves*" e "*another*" às vezes retornam efeitos voltados ao alvo errado.
- **Erros da LLM no parser:** extrapola tags (ex.: devolve `removal` para "*get rid of creatures*") e, nos testes, às vezes descartava termos essenciais do texto semântico (ex.: "*my*", ou "*wipe the board*", que gerou texto vazio). O prompt atual manda devolver ao texto semântico o que nenhum filtro captura, mas isso não foi reavaliado.
- **Nomes com erro de digitação:** o filtro de nome é por correspondência de texto, sem tolerância a erros. Nos testes, "*Pantlaza, Sun's Favored*" não encontrou "*Pantlaza, Sun-Favored*".
- **Embedding não treinado em *Magic*:** não conhece a hierarquia de tipos (*creatures* ⊂ *permanents*) nem jargão informal como "*go tall*".

## Próximos passos

- *Fine-tuning* do modelo de embedding com pares consulta em linguagem natural → cartas retornadas pelo Scryfall para a consulta equivalente em sintaxe.
- Melhorar o prompt e o *parsing* da LLM, com exemplos *few-shot* mais variados.
- Tolerância a erros de digitação na busca por nome.

# Professor orientador

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/jamesmalmeida">
        <img src="https://avatars.githubusercontent.com/u/108157661?v=4" width="100px;" alt="Foto do James no Github"/><br>
        <b>Prof. Dr. James Moraes de Almeida</b>
      </a>
    </td>
  </tr>
</table>

---

# Autores

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Giulio-Roux">
        <img src="https://avatars.githubusercontent.com/u/208799014?v=4" width="100px;" alt="Foto do Giulio no Github"/><br>
        <b>Giulio Oertel Spinelli Roux César</b>
      </a>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/JoaquimJFF">
        <img src="https://avatars.githubusercontent.com/u/208799542?v=4" width="100px;" alt="Foto do Joaquim no Github"/><br>
        <b>Joaquim Junior Ferola Fonseca</b>
      </a>
    </td>
  </tr>
</table>

## Como citar

```bibtex
@misc{roux2026mtgsearch,
  title  = {Buscador sem{\^a}ntico de cartas de Magic: The Gathering},
  author = {Roux C{\'e}sar, Giulio Oertel Spinelli and Fonseca, Joaquim Junior Ferola and Almeida, James Moraes de},
  year   = {2026},
  url    = {https://github.com/Giulio-Roux/MTG_Semantic_Searcher}
}
```

## Aviso legal

Este é um projeto acadêmico e não oficial. *Magic: The Gathering* é marca registrada da Wizards of the Coast LLC. Os dados de cartas vêm da API do [Scryfall](https://scryfall.com), que não é afiliado a este projeto.


