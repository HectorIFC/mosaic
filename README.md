# Mosaic

<p align="center">
  <img src="docs/logo/logo-512.png" alt="Mosaic" width="180">
</p>

> Lookup-based token embeddings for the JVM, in pure Kotlin.
>
> *Tessera é a peça. Mosaic é o todo.*

🌐 **Site:** [hectorifc.github.io/mosaic](https://hectorifc.github.io/mosaic/) (after Phase 5)

## Status

🚧 **Em desenvolvimento** — Fase 0 (Setup multi-módulo + infraestrutura)

Veja [PRD.md](./PRD.md) para a especificação completa.

## Sobre

Mosaic é uma **biblioteca Kotlin** que fornece uma `EmbeddingTable` treinável — uma matriz `[vocabSize × embeddingDim]` mapeando IDs de tokens a vetores densos de `Float`. Foi construída como projeto irmão do [Tessera](https://github.com/HectorIFC/tessera) (tokenizer BPE), completando o pipeline `texto → tokens → vetores` em Kotlin puro.

**O que Mosaic É:**
- Uma lookup table tipo `nn.Embedding` do PyTorch
- Storage eficiente em `FloatArray` 1D contíguo
- Operações vetoriais essenciais (cosine similarity, top-K nearest)
- Inicializadores plugáveis (uniform, Xavier, He, zeros, constant)
- Persistência binária compacta com checksum
- Integração nativa com Tessera

**O que Mosaic NÃO É:**
- Não treina embeddings auto-supervisionado (sem Word2Vec, sem backprop, sem SGD)
- Não faz operações matriciais densas (apenas vetor × vetor e matriz × vetor)
- Não acelera com GPU
- Não implementa quantização

A intenção é ser uma **peça de Lego** sólida: outros projetos (futuros ou externos) que queiram treinar embeddings de fato podem usar Mosaic como storage e atualizar os pesos via API.

### Princípios

- **Biblioteca, não aplicação** — destinada a ser consumida por outros projetos Kotlin
- **Kotlin puro** — sem libs de ML, sem libs de matemática (a menos que comprovadamente necessárias por performance)
- **`FloatArray` everywhere** — sem `Double`, sem boxing, sem `List<Float>`
- **Storage flat 1D** — cache locality é levado a sério
- **API pública minimalista** — só o necessário, marcado com `public` explicitamente

## Instalação (após release v0.0.1)

### Gradle (Kotlin DSL)

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}

// build.gradle.kts
dependencies {
    implementation("com.github.HectorIFC:mosaic:mosaic-core-v0.0.1")
    // Tessera vem como dependência transitiva — não precisa declarar explicitamente
}
```

## Uso básico

### Pipeline completo com Tessera

```kotlin
import dev.mosaic.EmbeddingTable
import dev.mosaic.TesseraEmbeddings
import dev.mosaic.Initializer
import dev.tessera.BpeTokenizer

fun main() {
    // 1. Carregar tokenizer (treinado previamente com Tessera)
    val tokenizer = BpeTokenizer.load("tessera.json")

    // 2. Criar embedding table com vocabSize compatível
    val embeddings = EmbeddingTable.create(
        vocabSize = tokenizer.vocabSize,
        embeddingDim = 128,
        initializer = Initializer.uniformDefault(seed = 42L)
    )

    // 3. Combinar em pipeline
    val pipeline = TesseraEmbeddings(tokenizer, embeddings)
    val vectors = pipeline.encode("Hello, mosaic!")
    
    println("Got ${vectors.size} vectors of dim ${vectors[0].size}")

    // 4. Salvar para reuso
    embeddings.save("mosaic.bin")
}
```

### Uso direto da EmbeddingTable

```kotlin
val table = EmbeddingTable.create(vocabSize = 1000, embeddingDim = 64)

// Lookup
val v = table.get(id = 42)

// Write (para treino externo)
val newVec = FloatArray(64) { 0.1f * it }
table.set(id = 42, vector = newVec)

// Similaridade
val similar = table.mostSimilar(id = 42, topK = 5)
similar.forEach { (id, score) ->
    println("Token $id → score $score")
}
```

Mais exemplos no módulo [`mosaic-samples`](./mosaic-samples/).

## Estrutura do projeto

Este é um projeto **Gradle multi-módulo** com 3 módulos:

```
mosaic/
├── mosaic-core/       ← A biblioteca (artefato publicado)
├── mosaic-cli/        ← Aplicação CLI consumindo a lib
└── mosaic-samples/    ← Exemplos de uso da lib
```

- **`mosaic-core`**: o JAR consumível. API pública minimalista.
- **`mosaic-cli`**: aplicação rodável (`./gradlew :mosaic-cli:run`) com comandos `create`, `inspect`, `stats`, `similar`, `encode`. Útil pra debug interativo de embeddings.
- **`mosaic-samples`**: pequenos programas Kotlin com `main()` mostrando padrões de uso da lib.

## Como rodar localmente

```bash
# Buildar tudo
./gradlew build

# Rodar os testes
./gradlew test

# Rodar a pipeline completa de qualidade
./gradlew test koverVerify ktlintCheck detekt

# Instalar a lib no Maven Local pra testar em outros projetos
./gradlew publishToMavenLocal

# Rodar a CLI
./gradlew :mosaic-cli:run --args="--help"
./gradlew :mosaic-cli:run --args="create --vocab-size 1000 --dim 64 --output embeddings.bin"
./gradlew :mosaic-cli:run --args="inspect --input embeddings.bin"

# Rodar um sample
./gradlew :mosaic-samples:run -PmainClass=dev.mosaic.samples.QuickStartSampleKt
```

## Arquitetura

Em alto nível:

1. **Storage:** A matriz é armazenada como `FloatArray` 1D contíguo de tamanho `vocabSize * embeddingDim`. Acesso a `row[i]` é via `data[i * dim .. (i+1) * dim - 1]`.
2. **Lookup:** `get(id)` retorna uma **cópia** do trecho — nunca uma referência interna.
3. **Operações vetoriais:** Todas em `VectorOps` (stateless). Acumulação em `Double` pra evitar perda de precisão; retorno em `Float`.
4. **`mostSimilar`:** Implementado com min-heap de tamanho K para O(N log K).
5. **Persistência:** Binário compacto (`.bin`) + metadata JSON (`.meta.json`). Checksum SHA-256 verifica integridade.
6. **Integração Tessera:** `TesseraEmbeddings` combina tokenizer e embedding numa única classe, validando vocabSize no construtor.

Veja [ARCHITECTURE.md](./ARCHITECTURE.md) (criado na Fase 4) para detalhes técnicos.

## Roadmap

- [x] Definir escopo e arquitetura (ver PRD.md)
- [ ] **Fase 0**: Setup Gradle multi-módulo + infraestrutura GitHub (workflows, dependabot, PR template, detekt) + Tessera dependency
- [ ] **Fase 1**: Core lib (EmbeddingTable, Initializer, VectorOps, mostSimilar)
- [ ] **Fase 2**: Persistência binária + integração TesseraEmbeddings
- [ ] **Fase 3**: Samples + benchmarks + cobertura ≥ 80%
- [ ] **Fase 4**: CLI (create, inspect, stats, similar, encode)
- [ ] **Fase 5**: GitHub Pages no ar (logo + vídeos MP4 + paleta laranja/preto/branco)
- [ ] **Fase 6**: Publicação no JitPack + polish

## Projetos relacionados

- **[Tessera](https://github.com/HectorIFC/tessera)** — tokenizer BPE byte-level (dependência de Mosaic)
- **Próximo projeto (futuro)** — pode ser um treinador de embeddings (Word2Vec/Skip-gram) ou um mini-transformer

## Licença

MIT — veja [LICENSE](./LICENSE).
