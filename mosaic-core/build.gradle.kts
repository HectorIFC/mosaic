plugins {
    kotlin("plugin.serialization")
    `maven-publish`
    `java-library`
    id("org.jetbrains.kotlinx.kover")
}

kotlin {
    explicitApi()
}

dependencies {
    api("com.github.HectorIFC:tessera:v0.0.6")
    api("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")

    testImplementation("io.kotest:kotest-runner-junit5:6.1.11")
    testImplementation("io.kotest:kotest-assertions-core:6.1.11")
}

kover {
    reports {
        verify {
            rule {
                bound {
                    minValue = 80
                }
            }
        }
    }
}

java {
    withSourcesJar()
    withJavadocJar()
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            groupId = "dev.mosaic"
            artifactId = "mosaic-core"
            version = project.version.toString()
            from(components["java"])
            pom {
                name.set("Mosaic Core")
                description.set("Lookup-based token embeddings for the JVM, in pure Kotlin.")
                url.set("https://github.com/HectorIFC/mosaic")
                licenses {
                    license {
                        name.set("MIT")
                        url.set("https://opensource.org/licenses/MIT")
                    }
                }
            }
        }
    }
    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/HectorIFC/mosaic")
            credentials {
                username = System.getenv("GITHUB_ACTOR") ?: project.findProperty("gpr.user") as String?
                password = System.getenv("GITHUB_TOKEN") ?: project.findProperty("gpr.key") as String?
            }
        }
    }
}
