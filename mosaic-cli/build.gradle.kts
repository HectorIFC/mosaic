plugins {
    application
}

dependencies {
    implementation(project(":mosaic-core"))

    testImplementation("io.kotest:kotest-runner-junit5:6.1.11")
    testImplementation("io.kotest:kotest-assertions-core:6.1.11")
}

application {
    mainClass.set("dev.mosaic.cli.MainKt")
}
