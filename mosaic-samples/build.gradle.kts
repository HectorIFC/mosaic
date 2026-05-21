plugins {
    application
}

dependencies {
    implementation(project(":mosaic-core"))
}

application {
    mainClass.set(
        project.findProperty("mainClass") as? String
            ?: "dev.mosaic.samples.QuickStartSampleKt",
    )
}
