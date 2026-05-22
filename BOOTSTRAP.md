# Guia de Bootstrap — Mosaic

Este guia descreve os passos que **você precisa executar na sua máquina** para colocar o projeto Mosaic no ar e começar a desenvolver com o Claude Code.

---

## Passo 1 — Criar o repositório no GitHub

### Opção A — Via web (mais simples)

1. Acesse https://github.com/new
2. Preencha:
   - **Repository name:** `mosaic`
   - **Description:** `Lookup-based token embeddings for the JVM, in pure Kotlin. Sister project to Tessera.`
   - **Visibility:** Public (recomendado, mesma escolha do Tessera)
   - **NÃO** marque "Add a README"
   - **NÃO** adicione `.gitignore` nem licença ainda
3. Clique em **Create repository**
4. Copie a URL do repositório

### Opção B — Via GitHub CLI

```bash
gh repo create mosaic \
  --public \
  --description "Lookup-based token embeddings for the JVM, in pure Kotlin. Sister project to Tessera." \
  --clone
```

### Topics sugeridos pro repo

Após criar, vai em **Settings → Topics** ou no painel direito **About → ⚙️** e adiciona:

```
kotlin embeddings nlp llm machine-learning jvm vector-embeddings kotlin-library educational from-scratch tessera word-embeddings token-embeddings
```

---

## Passo 2 — Clonar localmente

```bash
cd ~/projects
git clone https://github.com/HectorIFC/mosaic.git
cd mosaic
```

---

## Passo 3 — Adicionar os arquivos iniciais

Copie os dois arquivos que o Claude (chat) te entregou pra dentro do repositório:

```bash
# Dentro do diretório mosaic/
cp /caminho/onde/voce/salvou/PRD-Mosaic.md ./PRD.md
cp /caminho/onde/voce/salvou/README-Mosaic.md ./README.md
```

Faça o primeiro commit:

```bash
git add PRD.md README.md
git commit -m "docs: add initial PRD and README"
git push origin main
```

---

## Passo 4 — Abrir o Claude Code no projeto

```bash
cd ~/projects/mosaic
claude
```

---

## Passo 5 — Primeira mensagem ao Claude Code

Cole isso como sua primeira mensagem no Claude Code:

> Olá! Este é o projeto **Mosaic**, uma biblioteca Kotlin de embeddings (lookup table treinável). É o projeto irmão do Tessera (tokenizer BPE) que já está completo em `https://github.com/HectorIFC/tessera` (e também disponível localmente em `/Users/hectorcardoso/tessera`).
>
> Antes de qualquer ação:
>
> 1. Leia o arquivo `PRD.md` por completo. Ele contém escopo, decisões arquiteturais já tomadas (NÃO re-debater), plano de fases (6 fases: 0-6), critérios de aceitação e armadilhas conhecidas.
> 2. Leia também o `README.md` para contexto adicional.
> 3. **Tessera é a referência canônica.** Antes de criar workflows, configs, README, ou estrutura de `docs/`, consulte os equivalentes no Tessera e adapte. Mude apenas: nome (`tessera` → `mosaic`), paleta de cores no site (roxo/verde → laranja/preto/branco), nomes de comandos no CLI.
> 4. Em particular, atente para os pontos NÃO-NEGOCIÁVEIS:
>    - É uma **biblioteca**, não aplicação (mas com módulo CLI separado)
>    - **NÃO implementar treinamento auto-supervisionado** (sem Word2Vec, sem backprop)
>    - **Storage flat 1D obrigatório** (`FloatArray` único, não `Array<FloatArray>`)
>    - **Tessera é dependência via JitPack** — não reimplementar tokenização
>    - **3 módulos**: `mosaic-core` (lib), `mosaic-cli` (aplicação), `mosaic-samples` (exemplos)
>    - **Infraestrutura completa desde a Fase 0** — workflows CI/release, dependabot, PR template, detekt config
>    - **Site público** em `docs/` com paleta laranja/preto/branco (Fase 5)
> 5. Quando terminar, me apresente um **resumo do seu entendimento** do projeto, destacando: o que vamos construir, as decisões arquiteturais que você NÃO deve re-debater, e qual é a Fase 0 que vamos iniciar.
> 6. Aguarde minha confirmação antes de começar a implementar.
>
> Importante: siga estritamente as convenções do PRD (Conventional Commits com escopo, uma fase por vez, status report ao final de cada fase, mesmas ferramentas de qualidade do Tessera).

---

## Passo 6 — Durante o desenvolvimento

- **Trabalhe uma fase por vez.** Não deixe o Claude Code pular fases.
- **Revise os commits.** Periodicamente faça `git log --oneline`.
- **Rode os testes.** Após cada fase, `./gradlew test` deve passar 100%.
- **Sempre teste a publicação local.** Quando mudar API pública, rode `./gradlew publishToMavenLocal` e verifique que os samples ainda compilam consumindo a versão local.

---

## Estrutura final esperada do repositório (após Fase 0)

```
mosaic/
├── PRD.md
├── README.md
├── CHANGELOG.md
├── LICENSE
├── .gitignore
├── .editorconfig
│
├── .github/
│   ├── dependabot.yml
│   ├── pull_request_template.md
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
│
├── config/detekt/
│   └── detekt.yml
│
├── docs/                                # vazio na Fase 0; populado na Fase 5
│   └── (placeholder)
│
├── build.gradle.kts
├── settings.gradle.kts
├── gradle.properties
├── gradle/wrapper/
├── gradlew
├── gradlew.bat
│
├── mosaic-core/
│   ├── build.gradle.kts
│   └── src/
│       ├── main/kotlin/dev/mosaic/
│       │   └── (placeholders ou esqueleto)
│       └── test/kotlin/dev/mosaic/
│           └── HelloWorldTest.kt
├── mosaic-cli/
│   ├── build.gradle.kts
│   └── src/main/kotlin/dev/mosaic/cli/
│       └── Main.kt (placeholder com --help básico)
└── mosaic-samples/
    ├── build.gradle.kts
    └── src/main/kotlin/dev/mosaic/samples/
        └── (vazio na Fase 0)
```

---

## Diferenças importantes vs setup do Tessera

Ao contrário do Tessera, em Mosaic:

1. **Mesma estrutura de 3 módulos** (`mosaic-core` + `mosaic-cli` + `mosaic-samples`), mas o CLI tem comandos diferentes (foco em inspeção/debug de embeddings, não em treino/encode/decode de tokens).
2. **Dependência externa via JitPack:** `mosaic-core` declara `api("com.github.HectorIFC:tessera:tessera-core-v0.0.6")`.
3. **Tipos de arquivo de persistência diferentes:** Mosaic usa binário (`.bin`) + metadata JSON, não JSON puro.
4. **Sem corpus folder:** Mosaic não treina nada, não precisa de corpus.

---

## Checklist pessoal antes de começar

- [ ] Repositório criado no GitHub (`HectorIFC/mosaic`)
- [ ] Topics configurados
- [ ] Repositório clonado localmente
- [ ] `PRD.md` no root do projeto
- [ ] `README.md` no root do projeto
- [ ] Primeiro commit feito e pushado
- [ ] Claude Code aberto no diretório do projeto
- [ ] Primeira mensagem enviada

Quando todas as caixas estiverem marcadas, você está pronto pra começar a Fase 0.

Boa sorte! 🎨🧩 Tessera + Mosaic = pipeline completo.
