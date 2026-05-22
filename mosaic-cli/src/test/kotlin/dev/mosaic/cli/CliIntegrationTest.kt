package dev.mosaic.cli

import dev.tessera.Trainer
import dev.tessera.TrainingConfig
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.PrintStream
import java.nio.file.Files

// System.out / System.err are JVM-globals. If Kotest ever parallelizes specs
// (today it doesn't, but Kotest 6.x has parallelism switches), two `captured`
// calls running concurrently would interleave each other's stdout — the
// captured strings would be nondeterministic and assertions would flake.
// A single static lock serializes all captures across the whole test run.
private val ioCaptureLock = Any()

private fun captured(block: () -> Int): Triple<Int, String, String> = synchronized(ioCaptureLock) {
    val outBuf = ByteArrayOutputStream()
    val errBuf = ByteArrayOutputStream()
    val originalOut = System.out
    val originalErr = System.err
    System.setOut(PrintStream(outBuf, true, Charsets.UTF_8))
    System.setErr(PrintStream(errBuf, true, Charsets.UTF_8))
    try {
        val code = block()
        Triple(code, outBuf.toString(Charsets.UTF_8), errBuf.toString(Charsets.UTF_8))
    } finally {
        System.setOut(originalOut)
        System.setErr(originalErr)
    }
}

private fun cli(vararg args: String): Triple<Int, String, String> = captured { runCli(arrayOf(*args)) }

class CliIntegrationTest : StringSpec({

    val tmpRoot = Files.createTempDirectory("mosaic-cli-test").toFile()
    val binPath = File(tmpRoot, "table.bin").absolutePath
    val tokenizerPath = File(tmpRoot, "tokenizer.json").absolutePath
    val vocabSizeRef = IntArray(1)

    beforeSpec {
        // Train a small tokenizer once for the whole spec (used by similar/encode tests).
        val corpus = buildString {
            repeat(30) { append("the quick brown fox jumps over the lazy dog hello world 1234 ") }
        }
        val tokenizer = Trainer(TrainingConfig(numMerges = 50, verbose = false)).train(corpus)
        tokenizer.save(tokenizerPath)
        vocabSizeRef[0] = tokenizer.vocabSize
    }

    afterSpec {
        tmpRoot.deleteRecursively()
    }

    "no args prints global help" {
        val (code, out, _) = cli()
        code shouldBe 0
        out shouldContain "Mosaic CLI"
        out shouldContain "create"
        out shouldContain "inspect"
    }

    "--help prints global help" {
        val (code, out, _) = cli("--help")
        code shouldBe 0
        out shouldContain "Mosaic CLI"
    }

    "unknown command returns exit 1 and lists commands on stderr" {
        val (code, _, err) = cli("nope")
        code shouldBe 1
        err shouldContain "Unknown command: nope"
        err shouldContain "create"
    }

    "create --help works" {
        val (code, out, _) = cli("create", "--help")
        code shouldBe 0
        out shouldContain "Create a new embedding table"
        out shouldContain "--vocab-size"
    }

    "create writes a usable .bin file" {
        val (code, out, _) = cli(
            "create",
            "--vocab-size", "50",
            "--dim", "4",
            "--initializer", "uniform",
            "--seed", "1",
            "--output", binPath,
        )
        code shouldBe 0
        out shouldContain "saved to"
        File(binPath).exists() shouldBe true
        File("$binPath.meta.json").exists() shouldBe true
    }

    "create with missing required args returns exit 1" {
        val (code, _, err) = cli("create", "--dim", "4", "--output", "/tmp/x.bin")
        code shouldBe 1
        err shouldContain "Missing required argument"
    }

    "create with invalid initializer returns exit 1" {
        val (code, _, err) = cli(
            "create",
            "--vocab-size", "10",
            "--dim", "4",
            "--initializer", "magic",
            "--output", File(tmpRoot, "bad.bin").absolutePath,
        )
        code shouldBe 1
        err shouldContain "Unknown initializer"
    }

    "create constant requires --value" {
        val (code, _, err) = cli(
            "create",
            "--vocab-size", "10",
            "--dim", "4",
            "--initializer", "constant",
            "--output", File(tmpRoot, "bad-const.bin").absolutePath,
        )
        code shouldBe 1
        err shouldContain "constant"
        err shouldContain "--value"
    }

    "create accepts negative --value for constant initializer" {
        val target = File(tmpRoot, "negconst.bin").absolutePath
        val (code, _, _) = cli(
            "create",
            "--vocab-size", "4",
            "--dim", "2",
            "--initializer", "constant",
            "--value", "-0.5",
            "--output", target,
        )
        code shouldBe 0
        File(target).exists() shouldBe true
    }

    "inspect prints metadata and reports checksum valid" {
        val (code, out, _) = cli("inspect", "--input", binPath)
        code shouldBe 0
        out shouldContain "Mosaic Embedding Table"
        out shouldContain "Vocab size:          50"
        out shouldContain "Embedding dim:       4"
        out shouldContain "valid"
    }

    "inspect on missing file returns exit 2" {
        val (code, _, err) = cli("inspect", "--input", "/nonexistent/path.bin")
        code shouldBe 2
        err shouldContain "not found"
    }

    "inspect on file with flipped byte reports invalid checksum and exits 2" {
        val target = File(tmpRoot, "corrupted.bin")
        cli(
            "create",
            "--vocab-size", "5",
            "--dim", "3",
            "--initializer", "zeros",
            "--output", target.absolutePath,
        )
        val bytes = target.readBytes()
        val flipAt = 16 + 2
        bytes[flipAt] = (bytes[flipAt].toInt() xor 0xFF).toByte()
        target.writeBytes(bytes)

        val (code, out, _) = cli("inspect", "--input", target.absolutePath)
        code shouldBe 2
        out shouldContain "INVALID"
    }

    "stats prints summary statistics" {
        val (code, out, _) = cli("stats", "--input", binPath)
        code shouldBe 0
        out shouldContain "Statistics for"
        out shouldContain "Mean row norm:"
        out shouldContain "Total values:    200" // 50 * 4
    }

    "similar by id prints top-K" {
        val (code, out, _) = cli(
            "similar",
            "--embeddings",
            binPath,
            "--id",
            "5",
            "--top-k",
            "3",
        )
        code shouldBe 0
        out shouldContain "Top 3 most similar to token 5"
    }

    "similar rejects out-of-range id" {
        val (code, _, err) = cli(
            "similar",
            "--embeddings",
            binPath,
            "--id",
            "9999",
        )
        code shouldBe 1
        err shouldContain "out of range"
    }

    "similar by text requires tokenizer" {
        val (code, _, err) = cli(
            "similar",
            "--embeddings",
            binPath,
            "--text",
            "hello",
        )
        code shouldBe 1
        err shouldContain "tokenizer"
    }

    "similar by text works end-to-end" {
        val tokenizerEmbeddings = File(tmpRoot, "tok-emb.bin").absolutePath
        // Build embeddings sized to the trained tokenizer vocab
        val mkCode = cli(
            "create",
            "--vocab-size", vocabSizeRef[0].toString(),
            "--dim", "8",
            "--initializer", "uniform",
            "--seed", "7",
            "--output", tokenizerEmbeddings,
        ).first
        mkCode shouldBe 0

        val (code, out, _) = cli(
            "similar",
            "--embeddings", tokenizerEmbeddings,
            "--tokenizer", tokenizerPath,
            "--text", "hello",
            "--top-k", "5",
        )
        code shouldBe 0
        out shouldContain "Top 5 most similar to 'hello'"
    }

    "encode produces JSON output" {
        val embFile = File(tmpRoot, "enc-emb.bin").absolutePath
        cli(
            "create",
            "--vocab-size", vocabSizeRef[0].toString(),
            "--dim", "4",
            "--initializer", "zeros",
            "--output", embFile,
        )
        val (code, out, _) = cli(
            "encode",
            "--embeddings", embFile,
            "--tokenizer", tokenizerPath,
            "--text", "hello world",
            "--format", "json",
        )
        code shouldBe 0
        out shouldContain "\"ids\":"
        out shouldContain "\"vectors\":"
    }

    "encode default pretty format" {
        val embFile = File(tmpRoot, "enc-pretty.bin").absolutePath
        cli(
            "create",
            "--vocab-size", vocabSizeRef[0].toString(),
            "--dim", "4",
            "--initializer", "zeros",
            "--output", embFile,
        )
        val (code, out, _) = cli(
            "encode",
            "--embeddings",
            embFile,
            "--tokenizer",
            tokenizerPath,
            "--text",
            "hi",
        )
        code shouldBe 0
        out shouldContain "Encoded 'hi'"
    }

    "encode csv format outputs one line per token" {
        val embFile = File(tmpRoot, "enc-csv.bin").absolutePath
        cli(
            "create",
            "--vocab-size", vocabSizeRef[0].toString(),
            "--dim", "3",
            "--initializer", "constant",
            "--value", "0.25",
            "--output", embFile,
        )
        val (code, out, _) = cli(
            "encode",
            "--embeddings", embFile,
            "--tokenizer", tokenizerPath,
            "--text", "hello",
            "--format", "csv",
        )
        code shouldBe 0
        val rows = out.trim().lines()
        rows.isNotEmpty() shouldBe true
        // Each row: id,v1,v2,v3 → 4 fields
        for (row in rows) row.split(",").size shouldBe 4
    }

    "encode with unknown format returns exit 1" {
        val embFile = File(tmpRoot, "enc-bad.bin").absolutePath
        cli(
            "create",
            "--vocab-size", vocabSizeRef[0].toString(),
            "--dim", "2",
            "--initializer", "zeros",
            "--output", embFile,
        )
        val (code, _, err) = cli(
            "encode",
            "--embeddings", embFile,
            "--tokenizer", tokenizerPath,
            "--text", "x",
            "--format", "yaml",
        )
        code shouldBe 1
        err shouldContain "Unknown --format"
    }

    "every subcommand supports --help" {
        for (cmd in listOf("create", "inspect", "stats", "similar", "encode")) {
            val (code, out, _) = cli(cmd, "--help")
            code shouldBe 0
            out shouldContain "Usage:"
        }
    }
})
