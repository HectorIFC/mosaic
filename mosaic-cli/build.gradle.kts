plugins {
    application
}

dependencies {
    implementation(project(":mosaic-core"))
}

application {
    mainClass.set("dev.mosaic.cli.MainKt")
}
