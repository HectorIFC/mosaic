# PRD — Mosaic

> **Biblioteca Kotlin de embeddings (lookup table treinável).**
>
> *Tessera é a peça. Mosaic é o todo.*
>
> Onde **Tessera** transforma texto em IDs de tokens, **Mosaic** transforma esses IDs em vetores num espaço semântico.

---

## 📋 Documento de requisitos para implementação assistida via Claude Code

Este PRD descreve o projeto completo, decisões já tomadas, escopo, plano de fases, critérios de aceitação e armadilhas conhecidas. Foi construído após discussão detalhada sobre arquitetura. **Leia tudo antes de começar.**

---

## 1. Contexto e Motivação

### 1.1. Quem sou eu e por que esse projeto existe

Sou um desenvolvedor estudando LLMs **fazendo do zero**. Mosaic é o **segundo projeto** de um pipeline de NLP em Kotlin puro:

1. **[Tessera](https://github.com/HectorIFC/tessera)** (concluído, v0.0.6) — tokenizer BPE byte-level: converte texto ↔ IDs de tokens
2. **Mosaic** (este projeto) — embeddings: converte IDs ↔ vetores densos
3. **Futuro projeto** (ainda não definido) — possivelmente um mini-transformer ou classificador que consuma embeddings de Mosaic

O fluxo completo fica:

```
"Olá mundo" → Tessera.encode() → [156, 234, 89] → Mosaic.lookup() → [[0.21, -0.45, ...], [0.88, 0.12, ...], [-0.33, 0.67, ...]]
```

### 1.2. Natureza do projeto: BIBLIOTECA, não aplicação

**Mosaic é uma biblioteca**, igual Tessera. Mesma filosofia:

- Artefato principal é um **JAR consumível por outros projetos Kotlin/JVM**
- API pública minimalista e estável
- Publicação no JitPack
- Versionamento semântico
- **Inclui CLI como módulo separado**, demonstrando uso da lib e facilitando inspeção/debug de embeddings sem precisar escrever código Kotlin a cada teste

### 1.3. Princípios não-negociáveis

- **Kotlin puro, JVM target.** Sem dependências de bibliotecas de ML (sem DJL, KInference, DL4J).
- **Sem bibliotecas de matemática externas inicialmente.** Sem `multik`, sem `kmath`. Stdlib only para operações vetoriais.
- **Exceção pragmática:** se uma operação específica for medida como gargalo crítico (ex: `mostSimilar` em vocab > 50k), **só aí** considerar `multik` — e com justificativa documentada.
- **Standard library do Kotlin** para a lógica de embeddings. Exceções permitidas: build system (Gradle), testes (JUnit/Kotest), serialização (`kotlinx.serialization`), e a própria **Tessera** como dependência de runtime.
- **Sem treino "real"** (sem backprop, sem Word2Vec, sem SGD). Mosaic é uma lookup table treinável — os pesos podem ser **inicializados, lidos, modificados externamente e persistidos**, mas a lib **não implementa treinamento auto-supervisionado**. Se um dia for necessário treinar, será outro projeto.

### 1.4. Decisões arquiteturais já tomadas (NÃO RE-DECIDIR)

Essas decisões já foram tomadas após análise. **Não as questione** — siga-as:

1. **Caminho A: Lookup table treinável.** Não é Word2Vec, não é LSA, não é nada que envolva otimização por gradiente. É uma matriz `[vocabSize × embeddingDim]` exposta via API limpa.
2. **`FloatArray`, não `DoubleArray`.** Embeddings padrão da indústria são float32. Metade da memória do double, precisão sobrada.
3. **Tessera como dependência.** Mosaic importa `com.github.HectorIFC:tessera:tessera-core-v0.0.6` (ou versão mais recente) via JitPack. Mosaic **não reimplementa tokenização**.
4. **Multi-módulo com 3 módulos.** `mosaic-core` (lib), `mosaic-cli` (aplicação CLI consumindo a lib), `mosaic-samples` (exemplos de código). Igual à estrutura do Tessera.
5. **Storage flat (1D), não 2D.** A matriz de embeddings é um `FloatArray` único de tamanho `vocabSize * embeddingDim`, acessado com offset calculado. É 2-3x mais rápido que `Array<FloatArray>` por cache locality.
6. **Inicialização padrão: distribuição uniforme `[-0.5/dim, +0.5/dim]`**. Igual nn.Embedding do PyTorch.
7. **Persistência cedo.** Save/load é requisito da Fase 1, não "depois".
8. **API pública minimalista.** Tudo que não é parte da API pública deve ser `internal`. Use `explicitApi()`.
9. **Consultar Tessera em caso de dúvida.** Quando a especificação for ambígua sobre estrutura de arquivos, configs, workflows, ou estilo de README, o **Tessera é a referência canônica**. Repositório: `https://github.com/HectorIFC/tessera`. Caminho local na máquina do desenvolvedor: `/Users/hectorcardoso/tessera`. **Não inventar** estruturas próprias — espelhar Tessera e adaptar nomes (`tessera` → `mosaic`, paleta roxa → laranja).

---

## 2. Escopo

### 2.1. Dentro do escopo (MUST HAVE)

#### Como biblioteca

- API pública estável e bem documentada
- Classe `EmbeddingTable` representando uma matriz de embeddings treinável
- Operações de **lookup**: `get(id)`, `get(ids)`, `getRow(id)`
- Operações de **escrita**: `set(id, vector)`, `update(id, transform)`
- Inicializadores plugáveis: uniforme, Xavier/Glorot, He, zero, constante
- Operações vetoriais essenciais: `cosineSimilarity`, `dotProduct`, `norm`, `normalize`
- `mostSimilar(id, topK)` — top-K vizinhos mais próximos por similaridade cosseno
- `mostSimilar(vector, topK)` — variante para vetor arbitrário
- Persistência: save/load em formato binário compacto + metadata JSON
- Integração com Tessera: classe `TesseraEmbeddings` que combina tokenizer + embedding
- Suite de testes unitários cobrindo todas as operações + edge cases

#### Samples (módulo separado)

- `QuickStartSample.kt` — criar, popular, fazer lookup
- `TesseraIntegrationSample.kt` — pipeline completo texto → tokens → vetores via Tessera + Mosaic
- `SimilaritySample.kt` — explorar similaridades entre vetores
- `PersistenceSample.kt` — save/load com integridade verificada
- `InitializationSample.kt` — comparar inicializadores

#### Como aplicação CLI (módulo separado)

CLI mínima demonstrando uso da biblioteca e facilitando debug. Comandos:

- `create` — cria uma nova embedding table com dimensões e initializer especificados
- `inspect` — mostra metadata e estatísticas de um arquivo de embedding
- `similar` — top-K tokens mais próximos a um token ou texto (requer tokenizer + embedding)
- `encode` — pipeline completo texto → vetores (output em formato legível)
- `stats` — estatísticas detalhadas de uma matriz (min, max, mean, std, norma média)

A CLI **apenas chama a API pública** de `mosaic-core` — nenhum acesso a internals. Serve como exemplo "vivo" de consumo da lib.

#### Infraestrutura de repositório (espelhando Tessera)

- GitHub Actions workflows em `.github/workflows/`: `ci.yml` (tests + quality) e `release.yml` (versionamento automático via Conventional Commits, publicação)
- Dependabot configurado em `.github/dependabot.yml` (Gradle + GitHub Actions, weekly)
- Pull request template em `.github/pull_request_template.md`
- Configuração detekt em `config/detekt/detekt.yml` (thresholds e exceções adaptados ao projeto)
- ktlint via plugin Gradle, sem arquivo de config (usa defaults com `.editorconfig`)
- README principal seguindo o mesmo estilo visual e estrutura do Tessera

#### GitHub Pages (docs/)

- Site estático em `docs/` (servido via GitHub Pages, branch main, pasta `/docs`)
- `index.html` single-page com hero, features, how-it-works, visualizações, quick start, quality, footer
- **Paleta de cores diferente do Tessera**: tons de laranja, preto e branco (em vez de roxo/verde)
- 2 vídeos MP4 mostrando o pipeline visualmente (`lookup.mp4` e `similar.mp4`)
- Logo do projeto exibido na navbar e no hero
- Site webmanifest para PWA
- Estrutura `docs/logo/` com todos os tamanhos de logo + favicons

### 2.2. Fora do escopo (NÃO FAZER)

- **Treinamento auto-supervisionado** (Word2Vec, GloVe, FastText) — outro projeto
- Backpropagation, gradientes, otimizadores (SGD, Adam, etc)
- Embeddings contextuais (BERT, GPT-style)
- GPU / aceleração de hardware
- API REST / servidor
- Embeddings de imagem, áudio, etc — só texto
- Quantização (int8, int4)
- Distillation, pruning, ou qualquer forma de compressão
- Multiplatform (Kotlin/JS, Native) — só JVM por enquanto
- **Operações matriciais densas** (multiplicação matriz × matriz). Só matriz × vetor, e operações vetor × vetor.

### 2.3. Stretch goals (NICE TO HAVE, só se sobrar tempo após Fase 6)

- Carregamento de embeddings pré-treinados em formato word2vec/GloVe (read-only)
- Visualização básica via redução de dimensionalidade (PCA caseiro)
- Mini-benchmarks de performance com diferentes tamanhos de vocab e dim
- Implementação de `mostSimilar` com aproximação (FAISS-like, mas básico)

---

## 3. Definição de "Pronto" (Done)

O projeto é considerado finalizado quando **todos** estes critérios forem atendidos:

### 3.1. Critérios funcionais

- [ ] `lookup` retorna o vetor correto para qualquer ID válido
- [ ] `set` + `get` round-trip preserva valores exatamente (igualdade `Float`)
- [ ] Save + load preserva a matriz exatamente (todos os floats idênticos byte-a-byte)
- [ ] `cosineSimilarity(v, v) ≈ 1.0` para qualquer vetor não-nulo (tolerância 1e-6)
- [ ] `mostSimilar(id)` retorna o próprio ID como mais similar quando incluído
- [ ] Integração com Tessera funciona end-to-end (texto → vetores) sem erro
- [ ] IDs fora do range (negativo ou ≥ vocabSize) geram exceção clara

### 3.2. Critérios de qualidade

- [ ] Cobertura de testes ≥ 80% no módulo core (medir com `kover`, igual Tessera)
- [ ] Sem warnings do compilador Kotlin
- [ ] `explicitApi()` ativo e respeitado
- [ ] KDoc completo em toda API pública
- [ ] `mostSimilar(id, topK=10)` em vocab de 10.000 e dim=128 roda em menos de 100ms

### 3.3. Critérios de biblioteca

- [ ] Sample app consegue importar `mosaic-core` via Gradle e usar a API sem fricção
- [ ] JAR publicado via JitPack
- [ ] `mosaic-core` tem como dependências runtime apenas: `kotlin-stdlib`, `kotlinx-serialization-json`, e `tessera-core`
- [ ] Tudo que não é API pública está marcado como `internal`
- [ ] README com instruções de uso como **biblioteca**
- [ ] CLI funcional documentado no README do seu próprio módulo
- [ ] CLI **apenas usa API pública** de `mosaic-core` (verificado por inspeção)

### 3.4. Critérios de integração com Tessera

- [ ] Sample `TesseraIntegrationSample` demonstra pipeline completo funcionando
- [ ] Documentação explica como vocabSize deve casar com o tokenizer
- [ ] Não há reimplementação de tokenização em Mosaic

### 3.5. Critérios de infraestrutura e site (NOVOS — espelhando Tessera)

- [ ] `.github/workflows/ci.yml` espelha o do Tessera, verde no `main`
- [ ] `.github/workflows/release.yml` espelha o do Tessera, dispara em push para `main`
- [ ] `.github/dependabot.yml` configurado (Gradle + Actions, weekly)
- [ ] `.github/pull_request_template.md` no mesmo formato do Tessera (com `mosaic-core` no lugar de `tessera-core`)
- [ ] `config/detekt/detekt.yml` presente (versão adaptada — ver seção 4.9.5)
- [ ] ktlint via plugin Gradle, sem warnings
- [ ] GitHub Pages no ar em `hectorifc.github.io/mosaic/`
- [ ] Paleta laranja/preto/branco aplicada (sem reusar paleta roxa/verde do Tessera)
- [ ] Logo do projeto exibido no header do site e no topo do README
- [ ] Dois vídeos MP4 (`lookup.mp4` e `similar.mp4`) tocando em loop no site
- [ ] `site.webmanifest` no padrão PWA
- [ ] README principal segue o mesmo estilo estrutural do Tessera

---

## 4. Especificação Técnica

### 4.1. Identidade do projeto

- **Nome:** Mosaic
- **Tagline:** "Lookup-based token embeddings for the JVM, in pure Kotlin."
- **Group ID:** `dev.mosaic` (ajustar de acordo com seu domínio/preferência)
- **Artifact ID core:** `mosaic-core`
- **Artifact ID samples:** `mosaic-samples`
- **Package base:** `dev.mosaic`
- **Repositório:** `https://github.com/HectorIFC/mosaic`
- **Relação com Tessera:** projeto irmão; Mosaic depende de Tessera, mas não vice-versa

### 4.2. Stack

- **Linguagem:** Kotlin 2.0+ (target JVM 17+)
- **Build:** Gradle multi-módulo com Kotlin DSL
- **Testes:** Kotest (consistência com Tessera)
- **Serialização:** `kotlinx.serialization` (JSON para metadata)
- **Quality tooling:** Mesmas do Tessera (`ktlint`, `detekt`, `kover`)
- **Dependência crítica:** `com.github.HectorIFC:tessera:tessera-core-v0.0.6` (ou mais recente)
- **Publicação:** `maven-publish` plugin do Gradle, configurado para JitPack

### 4.3. Estrutura do projeto (Gradle multi-módulo)

```
mosaic/                               # repo raiz
├── settings.gradle.kts               # define os módulos
├── build.gradle.kts                  # config compartilhada
├── gradle.properties                 # versão global
├── README.md                         # documentação principal
├── PRD.md
├── ARCHITECTURE.md                   # criado na Fase 6
├── BENCHMARKS.md                     # criado na Fase 3
├── CHANGELOG.md                      # mantido desde a Fase 0
├── LICENSE
├── .gitignore
├── .editorconfig                     # copiado do Tessera
│
├── .github/                          # automação GitHub
│   ├── dependabot.yml                # weekly updates (Gradle + Actions)
│   ├── pull_request_template.md      # template adaptado do Tessera
│   └── workflows/
│       ├── ci.yml                    # tests + coverage + quality
│       └── release.yml               # versionamento via conventional commits
│
├── config/detekt/
│   └── detekt.yml                    # adaptado do Tessera
│
├── docs/                             # GitHub Pages (site público)
│   ├── index.html                    # single-page (laranja/preto/branco)
│   ├── site.webmanifest              # PWA manifest
│   ├── lookup.mp4                    # animação do processo de lookup
│   ├── similar.mp4                   # animação do mostSimilar
│   └── logo/
│       ├── logo.svg
│       ├── logo-1024.png             # + 512, 256, 128, 64
│       ├── favicon.ico
│       └── favicon-{16,32,48,180,192,512}.png
│
├── mosaic-core/                      # 🎯 A BIBLIOTECA
│   ├── build.gradle.kts              # configurado pra publicação
│   ├── README.md
│   └── src/
│       ├── main/kotlin/dev/mosaic/
│       │   ├── EmbeddingTable.kt         # classe principal
│       │   ├── Initializer.kt            # interface + impls públicas
│       │   ├── VectorOps.kt              # operações vetoriais públicas
│       │   ├── TesseraEmbeddings.kt      # integração com Tessera
│       │   ├── EmbeddingFormat.kt        # constantes do formato de save/load
│       │   └── internal/
│       │       ├── FlatMatrix.kt         # storage flat 1D
│       │       ├── Persistence.kt
│       │       ├── TopKHeap.kt           # heap para mostSimilar eficiente
│       │       └── Validators.kt
│       └── test/kotlin/dev/mosaic/
│           ├── EmbeddingTableTest.kt
│           ├── InitializerTest.kt
│           ├── VectorOpsTest.kt
│           ├── PersistenceTest.kt
│           ├── TesseraIntegrationTest.kt
│           └── ApiContractTest.kt
│
├── mosaic-cli/                       # aplicação CLI consumindo a lib
│   ├── build.gradle.kts              # depende de mosaic-core
│   ├── README.md
│   └── src/main/kotlin/dev/mosaic/cli/
│       ├── Main.kt                   # entry point + dispatcher
│       ├── CreateCommand.kt
│       ├── InspectCommand.kt
│       ├── SimilarCommand.kt
│       ├── EncodeCommand.kt
│       └── StatsCommand.kt
│
└── mosaic-samples/                   # exemplos de uso
    ├── build.gradle.kts
    ├── README.md
    └── src/main/kotlin/dev/mosaic/samples/
        ├── QuickStartSample.kt
        ├── TesseraIntegrationSample.kt
        ├── SimilaritySample.kt
        ├── PersistenceSample.kt
        └── InitializationSample.kt
```

### 4.4. API pública (do módulo `mosaic-core`)

**Esta é a única superfície que outros projetos vão tocar. Pensa com carinho.**

```kotlin
package dev.mosaic

/**
 * A trainable embedding table mapping token IDs to dense float vectors.
 *
 * Use cases:
 *  - As a building block for ML pipelines (the vectors are typically learned
 *    by an external training loop that calls [set] or [update]).
 *  - As a lookup for pre-trained embeddings loaded via [load].
 *
 * Mosaic does NOT implement training. It provides the storage, lookup,
 * persistence, and basic vector operations on top of it.
 */
public class EmbeddingTable internal constructor(
    public val vocabSize: Int,
    public val embeddingDim: Int,
    private val data: FloatArray,
) {

    /** Returns a copy of the vector at [id]. Mutating the returned array does NOT affect the table. */
    public fun get(id: Int): FloatArray

    /** Returns copies of the vectors at the given [ids], in order. */
    public fun get(ids: IntArray): Array<FloatArray>

    /** Writes [vector] into the row at [id]. The vector is copied; the source array can be mutated freely afterwards. */
    public fun set(id: Int, vector: FloatArray)

    /** Applies [transform] to the row at [id] in place. Useful for external training loops. */
    public fun update(id: Int, transform: (FloatArray) -> FloatArray)

    /** Returns the top-K most similar IDs to [id] by cosine similarity, sorted descending. */
    public fun mostSimilar(id: Int, topK: Int = 10, includeSelf: Boolean = true): List<Similarity>

    /** Returns the top-K most similar IDs to [query] vector by cosine similarity, sorted descending. */
    public fun mostSimilar(query: FloatArray, topK: Int = 10): List<Similarity>

    /** Persists the table to [path]. Format: binary matrix + JSON metadata. */
    public fun save(path: String)

    public fun save(file: java.io.File)

    public companion object {
        /**
         * Creates a new table with the given dimensions, populated by [initializer].
         */
        public fun create(
            vocabSize: Int,
            embeddingDim: Int,
            initializer: Initializer = Initializer.uniformDefault()
        ): EmbeddingTable

        public fun load(path: String): EmbeddingTable
        public fun load(file: java.io.File): EmbeddingTable
    }
}

/**
 * Pair of (id, similarity score), returned by [EmbeddingTable.mostSimilar].
 */
public data class Similarity(public val id: Int, public val score: Float)

/**
 * Strategy for initializing embedding vectors.
 */
public fun interface Initializer {
    /** Fills [target] with initial values. Implementations should be deterministic given a seed. */
    public fun fill(target: FloatArray, row: Int)

    public companion object {
        /** Uniform distribution in [-0.5/dim, +0.5/dim]. Default — matches PyTorch's nn.Embedding. */
        public fun uniformDefault(seed: Long = 42L): Initializer

        /** Uniform distribution in [-bound, +bound]. */
        public fun uniform(bound: Float, seed: Long = 42L): Initializer

        /** Xavier/Glorot uniform — bound = sqrt(6 / (fanIn + fanOut)). */
        public fun xavier(fanIn: Int, fanOut: Int, seed: Long = 42L): Initializer

        /** He uniform — bound = sqrt(6 / fanIn). For ReLU networks. */
        public fun he(fanIn: Int, seed: Long = 42L): Initializer

        /** All zeros. Rarely useful for embeddings but available. */
        public fun zeros(): Initializer

        /** All values set to [value]. */
        public fun constant(value: Float): Initializer
    }
}

/**
 * Stateless vector operations on FloatArrays.
 *
 * All operations assume vectors have the same dimension. Length mismatches throw IllegalArgumentException.
 */
public object VectorOps {
    public fun dotProduct(a: FloatArray, b: FloatArray): Float
    public fun norm(v: FloatArray): Float
    public fun cosineSimilarity(a: FloatArray, b: FloatArray): Float

    /** Returns a new normalized vector. Original is not mutated. */
    public fun normalize(v: FloatArray): FloatArray

    /** Normalizes [v] in place. */
    public fun normalizeInPlace(v: FloatArray)
}

/**
 * Combines a Tessera tokenizer and a Mosaic embedding table into a single text → vectors pipeline.
 *
 * Throws IllegalStateException at construction if the tokenizer's vocabSize doesn't match the embedding table's.
 */
public class TesseraEmbeddings(
    private val tokenizer: dev.tessera.BpeTokenizer,
    public val embeddings: EmbeddingTable,
) {

    init {
        require(tokenizer.vocabSize == embeddings.vocabSize) {
            "Vocab size mismatch: tokenizer=${tokenizer.vocabSize}, embeddings=${embeddings.vocabSize}"
        }
    }

    /** Tokenizes [text] with Tessera, then returns the corresponding embedding vectors. */
    public fun encode(text: String): Array<FloatArray>

    /** Returns the mean vector of all tokens in [text]. A simple "sentence embedding" baseline. */
    public fun encodeMeanPooled(text: String): FloatArray
}
```

Exemplo de uso (deve funcionar como sample):

```kotlin
import dev.mosaic.EmbeddingTable
import dev.mosaic.TesseraEmbeddings
import dev.mosaic.Initializer
import dev.tessera.BpeTokenizer

fun main() {
    val tokenizer = BpeTokenizer.load("tessera.json")

    val embeddings = EmbeddingTable.create(
        vocabSize = tokenizer.vocabSize,
        embeddingDim = 128,
        initializer = Initializer.uniformDefault(seed = 42L)
    )

    val pipeline = TesseraEmbeddings(tokenizer, embeddings)
    val vectors = pipeline.encode("Hello, mosaic!")

    println("Got ${vectors.size} vectors of dim ${vectors[0].size}")

    embeddings.save("mosaic.bin")
}
```

### 4.5. Formato de persistência

A persistência usa **dois arquivos** (ou opcionalmente um zip que os contém):

- `<name>.bin` — matriz como `FloatArray` raw, little-endian
- `<name>.meta.json` — metadata serializado com `kotlinx.serialization`

#### Metadata JSON

```json
{
  "version": 1,
  "vocabSize": 50000,
  "embeddingDim": 128,
  "format": "float32-le",
  "byteOrder": "little-endian",
  "checksum": "sha256-hex-here",
  "createdAt": "2026-05-20T14:30:00Z",
  "tesseraCompatible": true
}
```

#### Binário

- Cabeçalho fixo de 16 bytes (magic number + versão)
- Sequência contígua de `vocabSize * embeddingDim` floats em little-endian
- Sem padding, sem compressão
- Total = 16 + `vocabSize * embeddingDim * 4` bytes

**Por que binário?** Embeddings podem ser grandes (100MB+) e JSON com array de floats fica 3-5x maior e lento de parsear. Binário direto é compacto, rápido e exato (sem perda de precisão por arredondamento decimal).

### 4.6. Storage interno: flat 1D vs 2D

Storage 1D é **obrigatório**, não opcional:

```kotlin
// ✅ CORRETO: storage flat
internal class FlatMatrix(val rows: Int, val cols: Int) {
    val data: FloatArray = FloatArray(rows * cols)

    fun getRow(row: Int, into: FloatArray): FloatArray {
        System.arraycopy(data, row * cols, into, 0, cols)
        return into
    }
}

// ❌ ERRADO: nested arrays causam cache miss e indireção desnecessária
class BadMatrix(rows: Int, cols: Int) {
    val data: Array<FloatArray> = Array(rows) { FloatArray(cols) }
}
```

**Por que importa:** `Array<FloatArray>` é um array de ponteiros. Cada acesso causa indireção e potencial cache miss. `FloatArray` único é um bloco contíguo de memória — operações vetoriais sobre ele são significativamente mais rápidas (2-3x em benchmarks). Pra embedding tables grandes, isso vira a diferença entre 50ms e 150ms numa busca top-K.

### 4.7. Algoritmo de `mostSimilar` (top-K)

Implementação naïve (ordenar tudo, pegar primeiros K) é O(N log N). Para vocab grande, custa caro.

**Algoritmo recomendado:** min-heap de tamanho K, percorre uma vez O(N log K):

```kotlin
// Pseudocódigo
1. Pre-compute norm of query vector
2. Init min-heap of size K
3. For each row in [0, vocabSize):
    a. Compute cosine similarity (use dot product / (norm_query * norm_row))
    b. If heap.size < K: push (id, sim)
    c. Else if sim > heap.peek().score: replaceTop(id, sim)
4. Extract heap, sort descending
```

Para vocab = 50k e topK = 10, isso roda em ~10-20ms em hardware comum.

**Otimização opcional (post v1):** pre-normalizar todas as rows na construção; assim cosine vira só dot product e fica ~2x mais rápido. Mas isso muda semântica (rows ficam pré-normalizadas), então deve ser opt-in.

### 4.8. CLI (módulo `mosaic-cli`)

A CLI demonstra uso da lib e oferece debug interativo de embeddings. **Não tem lógica própria** além de parsing de argumentos e formatação de output — toda funcionalidade vem da API pública de `mosaic-core`.

#### Comandos

##### `create`
Cria uma nova `EmbeddingTable` com inicializador escolhido e salva no disco.

```bash
mosaic-cli create \
  --vocab-size 50000 \
  --dim 128 \
  --initializer uniform \
  --seed 42 \
  --output embeddings.bin
```

Opções:
- `--vocab-size`: tamanho do vocabulário (obrigatório)
- `--dim`: dimensão dos vetores (obrigatório)
- `--initializer`: `uniform` (default), `xavier`, `he`, `zeros`, `constant`
- `--seed`: seed para inicializadores aleatórios (default 42)
- `--bound`: bound para uniform (default 0.5/dim)
- `--value`: valor constante (só para `constant`)
- `--output` / `-o`: caminho de saída (obrigatório)

##### `inspect`
Mostra metadata de um arquivo de embeddings.

```bash
mosaic-cli inspect --input embeddings.bin
```

Output esperado:
```
Mosaic Embedding Table
─────────────────────────────────
Version:        1
Vocab size:     50,000
Embedding dim:  128
Format:         float32-le
File size:      24.41 MB
Checksum:       a3f4b2e1... (valid ✓)
Created:        2026-05-20T14:30:00Z
Tessera-compatible: yes
```

##### `stats`
Estatísticas detalhadas dos valores na matriz.

```bash
mosaic-cli stats --input embeddings.bin
```

Output esperado:
```
Statistics for embeddings.bin
─────────────────────────────────
Total values:   6,400,000 (50,000 × 128)
Min:            -0.0039
Max:             0.0039
Mean:           -0.0000
Std dev:         0.0023
Mean row norm:   0.0257
Min row norm:    0.0190
Max row norm:    0.0312
```

##### `similar`
Top-K tokens mais próximos a um token (por ID) ou texto (via Tessera).

```bash
# Por ID
mosaic-cli similar \
  --embeddings embeddings.bin \
  --id 1234 \
  --top-k 10

# Por texto (requer tokenizer Tessera)
mosaic-cli similar \
  --embeddings embeddings.bin \
  --tokenizer tessera.json \
  --text "gato" \
  --top-k 10
```

Output esperado:
```
Top 10 similar to "gato" (token 1234):
─────────────────────────────────
  1. token 1234  ( "gato")     score: 1.0000
  2. token 5678  ( "cachorro") score: 0.8423
  3. token  890  ( "felino")   score: 0.7891
  ...
```

##### `encode`
Pipeline completo: texto → tokens → vetores. Output em formato legível ou JSON.

```bash
mosaic-cli encode \
  --tokenizer tessera.json \
  --embeddings embeddings.bin \
  --text "Olá mundo" \
  --format json
```

Opções:
- `--format`: `pretty` (default, mostra dim + primeiros valores), `json` (vetores completos como JSON array), `csv`

#### Implementação

- Parser de argumentos: **manual** (igual Tessera) ou `kotlinx-cli`. Decidir uma vez e seguir.
- Plugin Gradle: `application` para distribuição via `installDist`
- Entry point: `dev.mosaic.cli.MainKt`
- Erros formatados de forma amigável (não stacktraces crus em uso normal)
- Códigos de saída: `0` sucesso, `1` erro de uso (argumento inválido), `2` erro de runtime (arquivo não existe, etc)

#### Critério de qualidade para a CLI

- [ ] Cada comando tem `--help` mostrando uso
- [ ] `mosaic-cli` sem argumentos mostra lista de comandos disponíveis
- [ ] **Nenhum acesso a símbolos `internal`** — só API pública
- [ ] Testes de integração rodando os comandos end-to-end (input → output esperado)
- [ ] README do módulo `mosaic-cli` documenta cada comando com exemplos

### 4.9. Infraestrutura de repositório (espelhando Tessera)

Esta seção descreve **exatamente** quais arquivos de automação e config replicar do Tessera. Cada item tem uma referência canônica em `https://github.com/HectorIFC/tessera` — quando em dúvida, abra o arquivo lá e adapte trocando `tessera` por `mosaic` e ajustando paths.

#### 4.9.1. `.github/workflows/ci.yml`

Workflow de CI rodando em PRs para `main` e em pushes para feature branches (com `branches-ignore: [main]`). Dois jobs em paralelo:

**Job `test`** (Tests & Coverage):
- Checkout (`actions/checkout@v6`)
- Setup Java 21 Temurin (`actions/setup-java@v5`)
- Setup Gradle (`gradle/actions/setup-gradle@v6`) com `cache-read-only` se ref ≠ main
- Run: `./gradlew test`
- Run: `./gradlew koverVerify` (coverage ≥ 80%)
- Run: `./gradlew koverHtmlReport` (com `if: always()`)
- Upload artifact: relatório de coverage em `mosaic-core/build/reports/kover/html/`, retention 7 dias

**Job `quality`** (Code Quality):
- Mesmo setup (checkout + Java + Gradle)
- Run: `./gradlew ktlintCheck`
- Run: `./gradlew detekt`
- Upload artifact detekt em failure: `**/build/reports/detekt/`, retention 7 dias

Inclui `concurrency: ci-${{ github.ref }}` com `cancel-in-progress: true` para cancelar runs anteriores quando novo push chega.

> Referência: `.github/workflows/ci.yml` do Tessera

#### 4.9.2. `.github/workflows/release.yml`

Workflow de release rodando em push para `main`. Faz versionamento automático via Conventional Commits.

Estrutura essencial:
- `concurrency: group: release, cancel-in-progress: false` (releases nunca são canceladas)
- `permissions: contents: write, packages: write`
- `checkout` com `fetch-depth: 0` (histórico completo para parser de commits)
- Setup Java 21 Temurin + Gradle
- **Compute next version** via `mathieudutour/github-tag-action@v6.2` com `dry_run: true`
- **Abort if no version change** se `new_tag == previous_tag`
- **Abort if tag already exists** (proteção contra duplicação)
- **Update version in all files** via `sed`:
  - `gradle.properties`: `version=X.Y.Z`
  - `docs/index.html`: substitui badge `vX.Y.Z` no hero, snippet JitPack, footer
  - `README.md`: snippet JitPack
  - `mosaic-core/README.md`: snippet JitPack e Maven coordinate
- **Stage version bump commit** localmente (sem push ainda)
- **Build & verify**: `./gradlew build koverVerify`
- **Publish to GitHub Packages** com tratamento especial de erro 409 (artefato já publicado em run interrompido anterior)
- **Push version bump to main** (após sucesso, com `[skip ci]` na mensagem)
- **Create git tag** via `mathieudutour/github-tag-action@v6.2` (modo real, não dry-run)
- **Create GitHub Release** via `ncipollo/release-action@v1` com `makeLatest: true`

Regras de bump (squash-merge PR title como conventional commit):
- `feat!: / BREAKING CHANGE:` → major (`0.1.0 → 1.0.0`)
- `feat:` → minor (`0.0.1 → 0.1.0`)
- `fix: / build: / chore:` → patch (`0.1.0 → 0.1.1`)
- Default bump quando nada se aplica: `patch`

> Referência: `.github/workflows/release.yml` do Tessera

**Atenção**: o release do Tessera publica no **GitHub Packages**, não no JitPack. Apesar do README e docs falarem em JitPack, o JitPack pega tags do git automaticamente — então criar a tag (que o workflow faz) já dispara o build no JitPack. As duas publicações coexistem.

#### 4.9.3. `.github/dependabot.yml`

Atualizações semanais (segundas) de dependências. Dois ecossistemas:

```yaml
version: 2
updates:
  # GitHub Actions
  - package-ecosystem: github-actions
    directory: /
    schedule:
      interval: weekly
      day: monday
    labels:
      - dependencies
      - github-actions
    commit-message:
      prefix: "build(deps)"

  # Gradle
  - package-ecosystem: gradle
    directory: /
    schedule:
      interval: weekly
      day: monday
    labels:
      - dependencies
      - gradle
    commit-message:
      prefix: "build(deps)"
    groups:
      kotlin:
        patterns: ["org.jetbrains.kotlin*"]
      kotest:
        patterns: ["io.kotest*"]
      kotlinx:
        patterns: ["org.jetbrains.kotlinx*"]
```

> Referência: `.github/dependabot.yml` do Tessera (cópia direta, sem mudanças)

#### 4.9.4. `.github/pull_request_template.md`

Template de PR igual ao do Tessera, com seções:

- **Description** (descrição livre)
- **Type of change** (checkbox: feat, fix, refactor, test, docs, build, chore)
- **Public API changed?** (No / Yes — com bloco de diff before/after)
- **Checklist**:
  - [ ] `./gradlew test` passes locally
  - [ ] `./gradlew koverVerify` passes (coverage ≥ 80%)
  - [ ] `./gradlew ktlintCheck` passes
  - [ ] `./gradlew detekt` passes
  - [ ] `CHANGELOG.md` updated under `[Unreleased]`
  - [ ] New public symbols have KDoc
  - [ ] Everything that is not public API is marked `internal`
- **Tests added / modified** (lista)
- **Notes for the reviewer** (contexto, decisões, armadilhas)

> Referência: `.github/pull_request_template.md` do Tessera. Adaptar texto onde mencionar `tessera-core` para `mosaic-core`.

#### 4.9.5. `config/detekt/detekt.yml`

Configuração customizada do detekt. Base é o template do Tessera, com ajustes específicos pro Mosaic:

```yaml
build:
  maxIssues: 0

config:
  validation: true
  warningsAsErrors: false

complexity:
  active: true
  ComplexMethod:
    active: true
    threshold: 15        # Tessera usa 20 (BPE loop); Mosaic pode usar 15 (default)
  LongMethod:
    active: true
    threshold: 60
  LongParameterList:
    active: true
    functionThreshold: 8
  TooManyFunctions:
    active: true
    thresholdInFiles: 20
    thresholdInClasses: 15
    thresholdInObjects: 15
  CyclomaticComplexMethod:
    active: true
    threshold: 15

style:
  active: true
  MagicNumber:
    active: true
    excludes:
      - '**/mosaic-samples/**'
      - '**/mosaic-cli/**'    # CLI também tem inline numbers em formatters
      - '**/test/**'
    ignoreNumbers:
      - '-1'
      - '0'
      - '1'
      - '2'
      - '4'      # bytes per float
      - '256'    # vocab base size — para compatibilidade conceitual com Tessera
    ignoreConstantDeclaration: true
    ignoreCompanionObjectPropertyDeclaration: true
    ignoreAnnotation: true
    ignoreNamedArgument: true
    ignoreEnums: true
  MaxLineLength:
    active: true
    maxLineLength: 120
  WildcardImport:
    active: true
    excludeImports:
      - 'java.util.*'
      - 'kotlinx.coroutines.*'
  UnusedPrivateMember:
    active: true
  ReturnCount:
    active: true
    max: 4
  ThrowsCount:
    active: true
    max: 3

naming:
  active: true
  FunctionNaming:
    active: true
    functionPattern: '[a-zA-Z][a-zA-Z0-9]*'
    excludes:
      - '**/test/**'
      - '**/*Test.kt'
      - '**/*Spec.kt'

performance:
  active: true

potential-bugs:
  active: true

exceptions:
  active: true
  TooGenericExceptionCaught:
    active: true
    exceptionNames:
      - Error
      - Exception
      - Throwable
      - RuntimeException
    allowedExceptionNameRegex: '_|(ignore|expected).*'

comments:
  active: false  # KDoc enforced separately via explicitApi
```

> Referência: `config/detekt/detekt.yml` do Tessera. Diferenças: thresholds ligeiramente menores (Mosaic não tem loops complexos como o BPE merge), e magic numbers adaptados (`4` para bytes per float, em vez de `255`).

#### 4.9.6. ktlint

Aplicado via plugin Gradle `org.jlleitschuh.gradle.ktlint` (igual Tessera). **Sem arquivo de config explícito** — usa defaults com integração ao `.editorconfig`.

#### 4.9.7. `.editorconfig`

Cópia direta do Tessera. Define encoding UTF-8, indent de 4 espaços (Kotlin), final newline, etc. Suporta ktlint sem config adicional.

### 4.10. GitHub Pages (`docs/`)

Página estática mostrando o projeto, servida via GitHub Pages (settings → Pages → source: `main` branch, `/docs` folder).

#### 4.10.1. Diretrizes visuais

**Paleta de cores:** diferente do Tessera (que usa roxo/verde sobre fundo quase-preto). Mosaic usa **laranja, preto e branco**:

```css
:root {
  --bg:        #0a0a0a;       /* preto quase puro */
  --bg2:       #141414;       /* surface elevada */
  --surface:   rgba(255,255,255,0.04);
  --border:    rgba(255,255,255,0.08);
  --primary:   #fb923c;       /* laranja claro (Tailwind orange-400) */
  --primary2:  #f97316;       /* laranja médio (orange-500) */
  --accent:    #ea580c;       /* laranja escuro (orange-600) */
  --text:      #f5f5f5;       /* branco quente */
  --muted:     #737373;       /* cinza médio */
  --muted2:    #a3a3a3;
  --code-bg:   #171717;
  --radius:    12px;
  --radius-lg: 20px;
}
```

Glow blobs (decoração de fundo) também em tons de laranja:
- `blob-1`: `#f97316` (laranja médio), top-left, opacity 0.12
- `blob-2`: `#fb923c` (laranja claro), middle-right
- `blob-3`: `#ea580c` (laranja escuro), bottom-center

**Tipografia:** mesma do Tessera — Inter para texto, JetBrains Mono para code. Importadas via Google Fonts.

**Tema dark.** Mosaic, igual Tessera, é dark-first. Não precisa ter modo claro.

#### 4.10.2. Estrutura do `index.html`

Single-page com as seguintes seções, na ordem do Tessera:

1. **Nav** (sticky, blur background): logo + nome + links (Features, How it works, Quality, Quick start) + botão GitHub
2. **Hero**: logo grande (88x88), badge de versão, headline (sugestão: `"Embeddings,"` + gradient `"piece by piece."` — ou outra que case com a metáfora do projeto), descrição, dois botões (Get started + View on GitHub), stats bar
3. **Stats bar** (6 cards): tests count, line coverage, integration tests, lookup operations, ML dependencies (0), max dim suportada
4. **Features**: cards descrevendo recursos (lookup, initializers, mostSimilar, Tessera integration, persistence, etc)
5. **How it works**: 4 etapas descrevendo o pipeline (token ID → flat offset → FloatArray copy → vector)
6. **Visualized** (seção com os 2 vídeos MP4)
7. **Quick start**: snippet Gradle + snippet Kotlin de uso mínimo
8. **Tech stack**: ícones/logos das tecnologias usadas (Kotlin, Gradle, JUnit/Kotest, etc)
9. **Quality**: cards com métricas (cobertura, fuzz tests, etc)
10. **Footer**: links repetidos + agradecimento

#### 4.10.3. Vídeos animados (MP4)

Dois MP4s na raiz de `docs/`:

**`lookup.mp4`** — animação visual do processo de lookup:
1. Mostra um token ID entrando (ex: `1234`)
2. Visualiza o cálculo do offset: `1234 × 128 = 157952`
3. Mostra o byte range sendo destacado num FloatArray
4. Os 128 floats saindo como um vetor

**`similar.mp4`** — animação do `mostSimilar`:
1. Mostra um vetor query
2. Visualiza o cálculo de cosine similarity contra cada row da matriz
3. Min-heap recebendo as top-K
4. Resultado final ordenado descendente

**Especificações técnicas:**
- Resolução: 1280×720 (mesma proporção dos vídeos do Tessera, que estão em ~350KB cada)
- Duração: 8-15 segundos cada (loop suave, sem cortes abruptos)
- Codec: H.264 baseline ou main, MP4 container
- Tamanho alvo: 300-500KB cada (para carregamento rápido)
- Sem áudio (vídeos no site são `autoplay loop muted`)
- Pode ser gerado com Manim, After Effects, FFmpeg + frames PNG, ou ferramenta similar

**Decisão sobre quando criar os vídeos:** os MP4s ficam para a Fase 5 (docs). É aceitável a Fase 5 começar com placeholders dos vídeos e iterar visualmente.

#### 4.10.4. Logo e favicons

Pasta `docs/logo/` contendo (igual Tessera):
- `logo.svg` — versão vetorial principal
- `logo-1024.png`, `logo-512.png`, `logo-256.png`, `logo-128.png`, `logo-64.png` — versões raster
- `logo-alt.svg`, `logo-alt-512.png` — variante alternativa (opcional)
- `favicon.ico` — favicon clássico
- `favicon-16.png`, `favicon-32.png`, `favicon-48.png`, `favicon-180.png` (Apple touch), `favicon-192.png`, `favicon-512.png`

**O logo deve casar visualmente com o site:** se a paleta do site é laranja/preto/branco, o logo deve refletir isso. **Não usar o gradient roxo/verde do Tessera no logo do Mosaic.**

Sugestão de conceito: várias peças (tesserae) formando um padrão maior — referência direta à metáfora "tessera é peça, mosaic é o todo". Em laranja sobre fundo preto, formato quadrado.

#### 4.10.5. `site.webmanifest`

PWA manifest no padrão Tessera:

```json
{
  "name": "Mosaic",
  "short_name": "Mosaic",
  "description": "Lookup-based token embeddings for the JVM, in pure Kotlin.",
  "icons": [
    {
      "src": "logo/favicon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "logo/favicon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ],
  "theme_color": "#f97316",
  "background_color": "#0a0a0a",
  "display": "standalone",
  "start_url": "."
}
```

#### 4.10.6. Meta tags

No `<head>` do `index.html`, incluir (espelhando Tessera, adaptado):

- Open Graph (`og:title`, `og:description`, `og:image`, `og:url`)
- Twitter cards (`summary_large_image`)
- Description meta para SEO
- Preconnect para Google Fonts
- Theme color para mobile browsers (`#f97316`)
- Favicons em múltiplos tamanhos (links já listados acima)

---

## 5. Plano de Implementação em Fases

### Fase 0 — Setup multi-módulo + infraestrutura (estimativa: 3-4h)

**Objetivo:** Projeto compilando, com a estrutura multi-módulo correta (3 módulos), testes rodando, Tessera importado, **e toda a infraestrutura de CI/CD/qualidade no lugar desde o início**.

#### Build e módulos

- [ ] `gradle init` com Kotlin DSL
- [ ] Configurar `settings.gradle.kts` declarando os 3 módulos (`mosaic-core`, `mosaic-cli`, `mosaic-samples`)
- [ ] `build.gradle.kts` raiz com config compartilhada (Kotlin 2.0, JVM 17, ktlint, detekt, kover plugins)
- [ ] `mosaic-core/build.gradle.kts` com `maven-publish`, `java-library`, `explicitApi()`
- [ ] **Adicionar JitPack como repositório** e Tessera como dependência:
  ```kotlin
  repositories {
      mavenCentral()
      maven { url = uri("https://jitpack.io") }
  }
  dependencies {
      api("com.github.HectorIFC:tessera:tessera-core-v0.0.6")
  }
  ```
- [ ] `mosaic-cli/build.gradle.kts` depende de `mosaic-core`, aplica plugin `application` com `mainClass = "dev.mosaic.cli.MainKt"`
- [ ] `mosaic-samples/build.gradle.kts` depende de `mosaic-core`
- [ ] `gradle.properties` com versão `0.1.0-SNAPSHOT`

#### Qualidade e config

- [ ] `.gitignore` (seguir o modelo do Tessera, adaptado: também ignorar `*.mosaic.bin` e `*.mosaic.json` exceto fixtures de teste)
- [ ] `.editorconfig` (copiar do Tessera)
- [ ] `config/detekt/detekt.yml` (conforme seção 4.9.5 — versão adaptada do Tessera)
- [ ] ktlint configurado via plugin (sem arquivo separado)
- [ ] kover configurado com threshold de 80%

#### Infraestrutura GitHub

- [ ] `.github/workflows/ci.yml` (conforme seção 4.9.1)
- [ ] `.github/workflows/release.yml` (conforme seção 4.9.2)
- [ ] `.github/dependabot.yml` (conforme seção 4.9.3)
- [ ] `.github/pull_request_template.md` (conforme seção 4.9.4)

#### Documentação básica

- [ ] README inicial focado em **uso como biblioteca**, no mesmo estilo do Tessera
- [ ] `CHANGELOG.md` inicial (formato Keep a Changelog)
- [ ] `LICENSE` (MIT)

#### Validação

- [ ] Um teste "hello world" passando em `mosaic-core`
- [ ] **Teste de integração mínimo:** importar `BpeTokenizer` de Tessera num arquivo de teste, criar instância vazia, verificar que compila e roda. Isso valida o setup da dependência.
- [ ] Push pra branch faz CI rodar e passar
- [ ] Push pra `main` faz release workflow rodar (vai falhar ao calcular versão se não houver commit `feat:`/`fix:` ainda — está ok)

**Critério de saída:** `./gradlew build` roda sem erro em todos os 3 módulos. `./gradlew publishToMavenLocal` instala `mosaic-core` no Maven local com sucesso. `./gradlew :mosaic-cli:run --args="--help"` executa. CI verde no primeiro PR.

### Fase 1 — Core da lib (estimativa: 2-3 dias)

**Objetivo:** API pública estável do `mosaic-core`, com lookup, set, e operações vetoriais funcionais.

- [ ] `internal/Validators.kt`: validações de bounds, dimensões, IDs
- [ ] `internal/FlatMatrix.kt`: storage 1D com row access otimizado
- [ ] `VectorOps.kt`: `dotProduct`, `norm`, `cosineSimilarity`, `normalize`, `normalizeInPlace`
- [ ] `Initializer.kt`: interface + companion com `uniformDefault`, `uniform`, `xavier`, `he`, `zeros`, `constant`
- [ ] `EmbeddingTable.kt`: classe principal com `get`, `set`, `update`, `create`
- [ ] `internal/TopKHeap.kt`: min-heap de tamanho fixo para top-K
- [ ] `EmbeddingTable.mostSimilar(id, topK)` e `mostSimilar(vector, topK)`
- [ ] Validação de IDs fora de range (negative, ≥ vocabSize) com mensagens claras

**Testes obrigatórios:**

- [ ] `set` + `get` round-trip preserva floats exatamente
- [ ] Inicializadores produzem valores no range esperado (testes estatísticos básicos com seed fixa)
- [ ] `cosineSimilarity(v, v) ≈ 1.0` para vetor não-nulo
- [ ] `cosineSimilarity(v, -v) ≈ -1.0`
- [ ] `cosineSimilarity(a, b) == cosineSimilarity(b, a)` (simetria)
- [ ] `mostSimilar(id)` inclui o próprio ID como primeiro quando `includeSelf=true`
- [ ] `mostSimilar` retorna lista ordenada descendente por score
- [ ] IDs inválidos lançam exceção apropriada
- [ ] Dimensões inválidas em `set` lançam exceção apropriada
- [ ] `ApiContractTest`: API pública compila e funciona conforme assinaturas declaradas

**Critério de saída:** Todos os testes passando. API pública completa e marcada com `public` + KDoc.

### Fase 2 — Persistência e integração com Tessera (estimativa: 1-2 dias)

**Objetivo:** Save/load funcionando + classe `TesseraEmbeddings` operacional.

- [ ] `EmbeddingFormat.kt`: constantes públicas (magic number, versão atual)
- [ ] `internal/Persistence.kt`: 
  - `writeBinary(path, table)` — escreve cabeçalho + floats em little-endian
  - `readBinary(path)` — lê e valida cabeçalho, retorna `FloatArray`
  - `writeMetadata(path, table)` — JSON com metadata
  - `readMetadata(path)` — parseia JSON
  - Cálculo de checksum SHA-256 para integridade
- [ ] `EmbeddingTable.save(path)` e `EmbeddingTable.load(path)`
- [ ] `TesseraEmbeddings.kt`:
  - Validação de vocabSize no construtor
  - `encode(text)` → tokeniza com Tessera, retorna `Array<FloatArray>`
  - `encodeMeanPooled(text)` → tokeniza, retorna vetor médio (sentence embedding baseline)

**Testes obrigatórios:**

- [ ] Save → load preserva matriz exatamente (todos os floats idênticos)
- [ ] Metadata inclui campos corretos
- [ ] Checksum detecta corrupção (modificar 1 byte do .bin → load falha)
- [ ] Versão incompatível no metadata gera exceção clara
- [ ] `TesseraEmbeddings` rejeita mismatch de vocabSize no construtor
- [ ] Pipeline completo: criar Tessera tokenizer + Mosaic embedding + `encode("texto")` retorna vetores válidos
- [ ] `encodeMeanPooled` retorna vetor de dimensão correta

**Critério de saída:** Pipeline texto → vetores funcionando end-to-end. Persistência verificada com checksum.

### Fase 3 — Samples e validação (estimativa: 1 dia)

**Objetivo:** Demonstrar que a biblioteca é consumível e funciona em casos reais.

- [ ] `QuickStartSample.kt`: criar tabela, fazer 5-10 operações básicas
- [ ] `TesseraIntegrationSample.kt`: 
  - Carregar tokenizer Tessera pré-treinado (ou treinar inline com corpus pequeno)
  - Criar embedding com vocabSize compatível
  - Encodar 3-5 frases diferentes
  - Mostrar dimensões e primeiros valores dos vetores
- [ ] `SimilaritySample.kt`: popular tabela com vetores conhecidos (manualmente), demonstrar `mostSimilar` retornando resultados esperados
- [ ] `PersistenceSample.kt`: criar, salvar, carregar, verificar igualdade
- [ ] `InitializationSample.kt`: criar 4 tabelas com inicializadores diferentes, mostrar estatísticas (min, max, média, desvio)
- [ ] `mosaic-samples/README.md` listando os samples
- [ ] Cobertura ≥ 80% verificada com `kover`
- [ ] `BENCHMARKS.md` com medições de:
  - Tempo de `mostSimilar(id, topK=10)` em vocab 10k, 50k, 100k
  - Tempo de save/load em diferentes tamanhos
  - Memória ocupada vs `vocabSize × embeddingDim × 4`

**Critério de saída:** Todos os samples rodam sem erro via `./gradlew :mosaic-samples:run`. Cobertura ≥ 80%. Benchmarks documentados.

### Fase 4 — CLI (estimativa: 1 dia)

**Objetivo:** Aplicação CLI demonstrando uso da lib em contexto real e oferecendo debug interativo.

- [ ] `mosaic-cli/src/main/kotlin/dev/mosaic/cli/Main.kt`: entry point com dispatcher de comandos e `--help` global
- [ ] Parser de argumentos (manual ou `kotlinx-cli` — decidir e seguir)
- [ ] `CreateCommand.kt`: cria embedding table com initializer escolhido e salva
- [ ] `InspectCommand.kt`: mostra metadata e validação de checksum
- [ ] `StatsCommand.kt`: estatísticas detalhadas (min/max/mean/std/norms)
- [ ] `SimilarCommand.kt`: top-K similares por ID ou texto
- [ ] `EncodeCommand.kt`: pipeline completo texto → vetores
- [ ] Formatação consistente de output (humanos primeiro)
- [ ] Tratamento de erros amigável (sem stacktrace cru em uso normal)
- [ ] Códigos de saída corretos (0/1/2)
- [ ] Distribuição via plugin `application` (`./gradlew :mosaic-cli:installDist`)
- [ ] `mosaic-cli/README.md` documentando cada comando com exemplos

**Testes obrigatórios:**

- [ ] Cada comando tem teste de integração rodando end-to-end
- [ ] `--help` funciona em cada subcomando
- [ ] Comando sem argumentos lista comandos disponíveis
- [ ] Argumentos inválidos retornam exit code 1 com mensagem clara
- [ ] Arquivo inexistente retorna exit code 2 com mensagem clara
- [ ] Verificação estática: CLI não acessa nenhum símbolo `internal` de `mosaic-core`

**Critério de saída:** Todos os comandos funcionam end-to-end. O CLI **apenas chama a API pública** do `mosaic-core` — nenhum acesso a internals. Documentação completa.

### Fase 5 — GitHub Pages + assets visuais (estimativa: 1-2 dias)

**Objetivo:** Site público de marketing/documentação no ar, com logo, animações MP4 e identidade visual própria.

#### Logo e branding

- [ ] Criar logo do Mosaic na paleta **laranja/preto/branco** (não reusar a paleta roxa/verde do Tessera)
- [ ] Conceito sugerido: peças de mosaico formando um padrão maior — reforçando a metáfora "tessera é peça, mosaic é o todo"
- [ ] Exportar nos tamanhos padrão: SVG + PNG (1024, 512, 256, 128, 64)
- [ ] Gerar favicons: `.ico` + PNGs em 16/32/48/180/192/512
- [ ] Salvar tudo em `docs/logo/`

#### Site (HTML/CSS)

- [ ] Criar `docs/index.html` conforme estrutura da seção 4.10.2
- [ ] Aplicar paleta laranja/preto/branco (variáveis CSS conforme 4.10.1)
- [ ] Importar Inter + JetBrains Mono via Google Fonts
- [ ] Implementar todas as seções: nav, hero, stats, features, how-it-works, visualized, quick start, tech stack, quality, footer
- [ ] Adaptar tagline do hero (sugestão: "Embeddings, piece by piece" — laranja gradient na metade colorida)
- [ ] Stats numbers devem refletir métricas reais do projeto (testes count, coverage %, etc — buscar após Fase 4)
- [ ] Animação de count-up nos números, scroll fade-in nas seções (mesma técnica do Tessera)
- [ ] Responsivo (mobile-first ou ao menos com breakpoints como Tessera)

#### Vídeos animados (MP4)

- [ ] Criar `docs/lookup.mp4` (animação do processo de lookup, conforme 4.10.3)
- [ ] Criar `docs/similar.mp4` (animação do mostSimilar, conforme 4.10.3)
- [ ] Especificações: 1280×720, H.264, 8-15s loop, sem áudio, ~300-500KB cada
- [ ] Ferramentas sugeridas: Manim (Python), After Effects, FFmpeg + frames PNG renderizados, ou ferramenta similar
- [ ] Inserir `<video autoplay loop muted playsinline>` nos cards da seção "Visualized"

#### PWA + Meta tags

- [ ] `docs/site.webmanifest` conforme seção 4.10.5
- [ ] Meta tags Open Graph + Twitter card no `<head>` (seção 4.10.6)
- [ ] Preconnect para Google Fonts
- [ ] Theme color `#f97316` no meta

#### Deploy

- [ ] Configurar GitHub Pages: Settings → Pages → Source: `main` branch, `/docs` folder
- [ ] Verificar URL: `https://hectorifc.github.io/mosaic/` (resposta 200, sem assets quebrados)
- [ ] Atualizar campo "Website" do repositório no GitHub com a URL acima
- [ ] Confirmar que o release workflow consegue atualizar versão em `docs/index.html` (já está coberto na seção 4.9.2)

**Critério de saída:** Site no ar em `hectorifc.github.io/mosaic/` com layout consistente com o do Tessera mas em paleta diferente. Logo visível na nav e no hero. Os dois MP4s tocam em loop. Lighthouse score ≥ 90 em performance e ≥ 95 em accessibility.

### Fase 6 — Publicação e polish (estimativa: meio dia)

**Objetivo:** Primeira release pública (v0.0.1) e polimento final.

- [ ] README principal completo (instalação via Gradle, exemplo mínimo, links pros samples e CLI), no mesmo estilo do Tessera incluindo logo no topo
- [ ] `mosaic-core/README.md` específico do módulo
- [ ] `mosaic-cli/README.md` com todos os comandos documentados
- [ ] `mosaic-samples/README.md` listando samples
- [ ] `ARCHITECTURE.md` explicando: storage flat, escolha de Float, algoritmo de top-K, relação com Tessera, design do CLI
- [ ] KDoc completo em toda API pública
- [ ] Validar que `release.yml` está funcionando (commit `feat:` em main dispara release)
- [ ] Tag `v0.0.1` criada (manualmente ou via release workflow ao fazer merge final)
- [ ] Validar que JitPack pegou a tag: visitar `https://jitpack.io/#HectorIFC/mosaic`
- [ ] Badges no README: build status (CI), JitPack version, coverage (kover), license
- [ ] `CHANGELOG.md` finalizado para v0.0.1

**Critério de saída:** Projeto consumível externamente — outra pessoa adiciona `mosaic-core` como dependência num projeto Kotlin novo e usa a API em < 5 minutos seguindo só o README. Site público com link no header do repo. CI verde no `main`.

Snippet que deve funcionar pra terceiros:

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        maven { url = uri("https://jitpack.io") }
    }
}

// build.gradle.kts
dependencies {
    implementation("com.github.HectorIFC:mosaic:mosaic-core-v0.0.1")
}
```

---

## 6. Armadilhas Conhecidas (LER ANTES DE CODAR)

### 6.1. Float vs Double

`Float` (32-bit) é o padrão para embeddings (PyTorch, TF, todos). `Double` (64-bit) gasta o dobro de memória sem ganho prático. **Use `FloatArray` em todo lugar**, exceto onde precisão extra é crítica (acumulação em normas — fazer soma em `Double` e converter de volta pode evitar perda):

```kotlin
public fun dotProduct(a: FloatArray, b: FloatArray): Float {
    require(a.size == b.size) { "Dimension mismatch" }
    var sum = 0.0  // Double pra evitar perda em acumulação
    for (i in a.indices) sum += a[i].toDouble() * b[i].toDouble()
    return sum.toFloat()
}
```

### 6.2. NaN e Infinity

Operações vetoriais podem gerar NaN (norma zero, divisão por zero). **Decida explicitamente** o comportamento:

- `normalize(vetorZero)` — retorna vetorZero? Lança exceção? **Recomendo:** retornar uma cópia do próprio vetor zero (não lançar), e documentar.
- `cosineSimilarity` quando uma das normas é zero — **recomendo:** retornar 0.0f, documentar.

### 6.3. Cópia vs referência em `get`

A API pública retorna **cópias**, não referências internas. Isso é segurança e evita bugs sutis:

```kotlin
// ✅ CORRETO
public fun get(id: Int): FloatArray {
    val result = FloatArray(embeddingDim)
    System.arraycopy(data, id * embeddingDim, result, 0, embeddingDim)
    return result
}

// ❌ ERRADO: vazaria referência interna; mutação externa corromperia a tabela
public fun get(id: Int): FloatArray = data.copyOfRange(id * embeddingDim, (id + 1) * embeddingDim)
// (esse exemplo cria cópia também, mas conceitualmente: nunca retorne `data` diretamente)
```

Para operações de alta performance onde a cópia importa, expor um overload `get(id, into: FloatArray)` que escreve no buffer fornecido pelo caller.

### 6.4. Determinismo de seeds

Inicializadores usam `Random` com seed. **Sempre** aceite seed como parâmetro. Testes que dependem de valores específicos devem usar seeds fixas. Inicialização não determinística é pesadelo de debug em ML.

### 6.5. Boundary conditions em `mostSimilar`

- `topK = 0` → retornar lista vazia, sem exceção
- `topK ≥ vocabSize` → retornar todos os IDs
- `vocabSize == 0` → tabela "vazia" é válida ou inválida? **Recomendo:** inválida, validar no construtor (`require(vocabSize > 0)`)

### 6.6. Endianness na persistência

JVM é big-endian por padrão em algumas operações, mas **convencione little-endian** no formato (igual numpy, PyTorch). Use `ByteBuffer.order(ByteOrder.LITTLE_ENDIAN)` explicitamente. Senão, arquivos salvos em uma máquina podem não carregar em outra arquitetura.

### 6.7. Tamanho de arquivo é previsível, valide

Após carregar `.meta.json`, calcule o tamanho esperado do `.bin`:

```kotlin
val expectedBytes = HEADER_SIZE + (metadata.vocabSize.toLong() * metadata.embeddingDim * 4)
require(file.length() == expectedBytes) { "Corrupted .bin: expected $expectedBytes bytes, got ${file.length()}" }
```

Erro precoce > erro misterioso.

### 6.8. Memória para vocab grande

Vocab = 50.000 × dim = 768 × 4 bytes = ~150 MB. Realista. Mas vocab = 200.000 × dim = 4096 × 4 = ~3.3 GB — heap default do JVM não aguenta. Documentar isso. Não otimizar agora, mas mencionar no README.

### 6.9. Tessera dependency drift

Tessera ainda está em versão `0.x.y` — API pode mudar. **Pinne a versão exata** no `build.gradle.kts`:

```kotlin
api("com.github.HectorIFC:tessera:tessera-core-v0.0.6")  // ✅ versão exata
// não use:
api("com.github.HectorIFC:tessera:tessera-core-v0.0.+")  // ❌ range pode quebrar
```

Atualize manualmente quando Tessera lançar versão nova compatível.

### 6.10. `explicitApi()` quebra primeira build

Ao ativar `explicitApi()` no Kotlin DSL, **todo símbolo no módulo** precisa ter visibilidade explícita. A primeira build vai falhar com dezenas de erros. **Isso é esperado.** Corrige um por um, marcando o que for público e tornando `internal` o que não for.

### 6.11. `FloatArray` na assinatura pública é "puro"

`FloatArray` é tipo nativo do Kotlin (não é genérico boxado). Pode aparecer livremente em API pública sem warnings. **Já** `Array<FloatArray>` também é ok. Não confunda com `List<Float>` que faz boxing.

### 6.12. Operações in-place vs imutáveis

Forneça **ambas** quando faz sentido (`normalize` e `normalizeInPlace`). Mas seja explícito no nome — sufixo `InPlace` deixa claro que muta. **Default deve ser imutável** (criar novo array).

### 6.13. Workflow de release dispara em todo push pra main

O `release.yml` espelha o do Tessera e roda em **todo push para `main`**. Se você commitar direto na `main` (em vez de via PR), vai disparar um release. **Sempre** mergeie via PR — o workflow tem proteções (`Abort if no version change`, `Abort if tag already exists`) mas é melhor evitar.

### 6.14. `[skip ci]` na mensagem de commit é essencial

O release workflow faz `git push` de volta pro `main` com o version bump. Sem `[skip ci]` na mensagem, isso dispara **outro** release workflow, criando loop infinito. A mensagem deve ser exatamente: `chore: release vX.Y.Z [skip ci]`. Não esqueça do `[skip ci]`.

### 6.15. GitHub Pages tem cache

Depois de mergear mudanças em `docs/`, o site pode levar 1-5 minutos pra atualizar. Se o conteúdo não mudou, force refresh (Cmd+Shift+R / Ctrl+Shift+R). Não fique reembarcando achando que o deploy falhou.

### 6.16. MP4 muito grande quebra a página

Vídeos MP4 em `docs/` têm que ser otimizados. Tessera mantém ~350KB cada. Se passarem de 2MB, a página fica lenta pra carregar em mobile. Use FFmpeg pra otimizar:

```bash
ffmpeg -i input.mp4 -vcodec libx264 -crf 28 -preset slow -an -vf "scale=1280:720" output.mp4
```

### 6.17. Logo deve combinar com a paleta do site

Se o site é laranja/preto/branco e o logo é roxo/verde (reusado do Tessera por engano), o resultado é inconsistente. **Crie um logo novo** específico pro Mosaic na paleta correta. Não economize aqui — o logo aparece na nav, no hero, no README, e nos favicons.

### 6.18. Versão hardcoded em vários lugares

A versão `vX.Y.Z` aparece em pelo menos 4 lugares: `gradle.properties`, `docs/index.html`, `README.md`, `mosaic-core/README.md`. O `release.yml` atualiza todos via `sed`. Quando você adicionar a versão em outro arquivo novo, **adicione o `sed` correspondente** no workflow. Senão a versão fica defasada e quebra o link de instalação.

### 6.19. Dependabot vai abrir muito PR no começo

Logo após setup, dependabot pode abrir 10+ PRs (cada workflow action, cada lib Gradle). É normal. Faça merge dos confiáveis em batch, mas **valide CI** em pelo menos um antes de fazer merge de todos.

---

## 7. Recursos e Referências

### 7.1. Referências canônicas

- **PyTorch `nn.Embedding`:** https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html — referência de API
- **TensorFlow `tf.keras.layers.Embedding`:** referência alternativa
- **Word2Vec original paper:** Mikolov et al. 2013 — pra entender o contexto histórico (não pra implementar agora)
- **Storage flat vs nested:** "Data-Oriented Design" — princípio geral, vale pra qualquer matriz numérica

### 7.2. Referências de Kotlin/JVM

- **Tessera (projeto irmão):** https://github.com/HectorIFC/tessera — referência de estrutura, build, qualidade
### 7.2. Referências de Kotlin/JVM

- **Tessera (projeto irmão — REFERÊNCIA PRIMÁRIA):** `https://github.com/HectorIFC/tessera` ou localmente em `/Users/hectorcardoso/tessera`. Quando em dúvida sobre estrutura de arquivos, configs (detekt, ktlint, editorconfig), workflows GitHub, PR template, formato de README, estrutura do `docs/`, **consulte o Tessera primeiro**. Adapte trocando `tessera` por `mosaic` e ajustando a paleta de cores.
- **Kotlin API guidelines:** https://kotlinlang.org/docs/api-guidelines-introduction.html
- **`explicitApi` mode:** https://kotlinlang.org/docs/whatsnew14.html#explicit-api-mode-for-library-authors
- **JitPack guide:** https://jitpack.io/docs/BUILDING/

### 7.3. Background conceitual (opcional)

- **"Word2Vec Tutorial - The Skip-Gram Model":** Chris McCormick blog — pra entender por que embeddings têm essa forma
- **"What is the difference between embedding layers and word2vec?":** Cross Validated — pra entender o lugar exato de Mosaic no ecossistema

---

## 8. Workflow com Claude Code

### 8.1. Como você (Claude Code) deve operar

1. **Leia este PRD inteiro antes de qualquer ação.**
2. Confirme que entendeu o escopo e as decisões já tomadas. Em particular:
   - Mosaic é uma **biblioteca**, não aplicação
   - **Não implementar treinamento auto-supervisionado** (sem Word2Vec, sem backprop)
   - **Storage flat 1D obrigatório**
   - **Tessera é dependência** — não reimplementar tokenização
   - **Tessera é a referência canônica** para tudo relacionado a estrutura de repo, configs, workflows, README e site (consulte `https://github.com/HectorIFC/tessera` ou `/Users/hectorcardoso/tessera`)
   - **Paleta do site**: laranja/preto/branco — NÃO reuse a paleta roxa/verde do Tessera
3. Trabalhe **fase por fase**. Não pule fases.
4. Ao começar uma fase, mostre o plano específico antes de codar.
5. Commite frequentemente, mensagens descritivas. Cada subtarefa = 1 commit mínimo.
6. Rode os testes após cada mudança significativa.
7. Ao terminar uma fase, mostre os critérios de saída atingidos antes de prosseguir.
8. Sempre que mudar a API pública do `mosaic-core`, rode `publishToMavenLocal` e teste em `mosaic-samples`.
9. Se encontrar decisão não coberta pelo PRD, pergunte.

### 8.2. Convenções de código

- `explicitApi()` ativo no módulo core
- Imutabilidade por padrão (`val`, não `var`)
- Operações que mutam terminam com sufixo `InPlace`
- Funções pequenas (< 30 linhas idealmente)
- Nomes claros, em inglês
- Comentários explicando **por que**, não **o que**
- KDoc em **toda** API pública (obrigatório, não opcional)
- Use `require` para pre-conditions (valida argumentos do caller)
- Use `check` para invariantes (valida estado interno)

### 8.3. Convenções de git

Mesmo padrão do Tessera — Conventional Commits com escopo:
- `feat(core): add EmbeddingTable.mostSimilar`
- `feat(samples): add Tessera integration sample`
- `feat(core): add Xavier initializer`
- `fix(core): handle zero-norm vectors in cosineSimilarity`
- `test(core): add round-trip tests for persistence`
- `docs: update README with installation instructions`
- `refactor(core): extract FlatMatrix to internal package`
- `build: configure maven-publish for jitpack`
- `perf(core): use min-heap for mostSimilar (O(N log K))`

### 8.4. Versionamento

- `0.1.0-SNAPSHOT` → Fase 0
- `0.x.y` → fases 1-5 (API ainda instável)
- `0.0.1` → fim da Fase 6 (primeira release, seguindo padrão Tessera)
- `1.0.0` → quando API estiver provada em uso real, ainda longe

Após v1.0.0, mudanças breaking exigem bump de major.

---

## 9. Comunicação e Bloqueios

### 9.1. Quando perguntar ao usuário (eu)

- Decisões fora do escopo do PRD
- Trade-offs significativos de design (especialmente em API pública)
- Se eu insistir em algo que viola decisão fechada na seção 1.4, **questione antes de fazer**
- Resultados de benchmarks
- Ao final de cada fase

### 9.2. Quando NÃO perguntar

- Detalhes de implementação cobertos pelo PRD
- Escolhas estéticas de código
- Quais testes adicionar

### 9.3. Status report ideal ao final de cada fase

```
✅ Fase X concluída.

Implementado:
- item 1
- item 2

API pública mudou? Sim/Não. Se sim:
- diff resumido

Testes adicionados:
- N testes, todos passando

Critérios de saída:
- [x] critério A
- [x] critério B

Próximos passos: iniciando Fase Y. Posso prosseguir?
```

---

## 10. Apêndice — Glossário

- **Tessera:** o projeto irmão (tokenizer). Cada peça do mosaico.
- **Mosaic:** este projeto. O todo formado pelas peças no espaço vetorial.
- **Embedding:** representação densa de um token como vetor de floats.
- **Embedding table / lookup table:** matriz `[vocabSize × embeddingDim]` mapeando IDs a vetores.
- **VocabSize:** número de tokens no vocabulário (deve casar com o tokenizer).
- **EmbeddingDim:** dimensão de cada vetor (tipicamente 50-1024).
- **Cosine similarity:** medida de similaridade entre vetores, ignora magnitude, foca em direção. Range [-1, 1].
- **Top-K nearest:** os K IDs com maior similaridade a um dado vetor/ID.
- **Initializer:** estratégia de preenchimento inicial da matriz (uniforme, Xavier, He, etc).
- **Mean pooling:** combinar múltiplos vetores tirando média elemento a elemento. Sentence embedding baseline.
- **API pública:** símbolos `public` em `mosaic-core`, consumíveis por terceiros.
- **`internal`:** símbolos visíveis apenas dentro do módulo.
- **Storage flat:** matriz armazenada como `FloatArray` 1D contíguo, em vez de `Array<FloatArray>`.

---

## 11. Checklist mestre

- [ ] Fase 0: Setup multi-módulo completo (3 módulos), infraestrutura GitHub (workflows, dependabot, PR template, detekt), Tessera importada
- [ ] Fase 1: Core lib funcional com lookup, set, operações vetoriais e mostSimilar
- [ ] Fase 2: Persistência + integração TesseraEmbeddings
- [ ] Fase 3: Samples consumindo a lib, cobertura ≥ 80%, benchmarks
- [ ] Fase 4: CLI consumindo a lib, comandos `create/inspect/stats/similar/encode`
- [ ] Fase 5: GitHub Pages no ar, logo em laranja/preto/branco, vídeos MP4 (lookup + similar)
- [ ] Fase 6: Publicação no JitPack, READMEs polidos, ARCHITECTURE.md, tag `v0.0.1`
- [ ] Todos os critérios da seção 3 atingidos
- [ ] README focado em uso como biblioteca, no estilo do Tessera
- [ ] Site público em `hectorifc.github.io/mosaic/`
- [ ] JitPack badge verde

**Quando esse checklist estiver completo, Mosaic está pronta como biblioteca pública. O pipeline texto → tokens → vetores está disponível em Kotlin puro, ponta a ponta, via Tessera + Mosaic, com CLI para debug, site público com vídeos animados, e CI/CD/release automation completos.**

---

## 📜 Sobre o nome

> *"Mosaic"* — uma obra de arte composta por muitas peças individuais (tesserae) arranjadas para formar uma imagem completa.
>
> Em Tessera, cada token é uma peça isolada — um pedaço de bytes com um ID. Em Mosaic, essas peças encontram seu lugar num espaço vetorial onde, juntas, começam a formar significado. Um token sozinho é apenas um índice; em conjunto com seus vizinhos, é parte de uma estrutura semântica maior.
>
> A relação Tessera → Mosaic não é só nominal: é o caminho natural que qualquer pipeline de NLP percorre. Tokenizar, então embedar. A peça, então o todo.

---

## 12. Considerações para o futuro (post-v1.0)

Decisões que **não** precisam ser tomadas agora, mas vale antecipar:

- **Treinamento:** Se um dia for desejável treinar embeddings de verdade, será projeto separado (provavelmente chamado algo como `Tapestry` ou `Loom` — algo que evoque "tecer"). Ele teria como dependência tanto Tessera quanto Mosaic, e implementaria Word2Vec/Skip-gram alimentando os pesos via `EmbeddingTable.update()`.
- **Mini-Transformer:** O terceiro projeto natural depois de Mosaic seria um mini-transformer que consome embeddings e produz output via atenção. Aí o ecossistema estaria completo.
- **Multiplatform:** Se Mosaic ganhar tração, pode fazer sentido migrar pra Kotlin Multiplatform. A API atual é compatível com isso — só usar tipos JVM-only (`java.io.File`) com cuidado.

Mas tudo isso é depois. Foco no presente: v0.0.1, lookup table treinável, simples e bem feita.
