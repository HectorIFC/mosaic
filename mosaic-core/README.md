# mosaic-core

The Mosaic library — a lookup-based embedding table for the JVM, in pure Kotlin.

> **Status:** Phase 0 (setup). The public API is **not yet implemented**; this README is a placeholder so the release workflow can update version references. Full module documentation lands in Phase 6.

## Adding the dependency

### Gradle (Kotlin DSL) — via JitPack

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
    implementation("com.github.HectorIFC:mosaic:mosaic-core-v0.0.0")
}
```

### Gradle (Kotlin DSL) — via GitHub Packages

```kotlin
dependencies {
    implementation("dev.mosaic:mosaic-core:0.0.0")
}
```

> The `v0.0.0` / `0.0.0` placeholders above are rewritten by the release workflow once the first real version is published.
