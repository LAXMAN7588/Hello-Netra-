package com.hellonetra.ocr

import android.util.Log
import com.paddle.ocr.model.OCRResult
import java.util.Locale

/*
 * English Text Normalization for TTS
 *
 * Adapted in part from open-source English text normalization rules from:
 * - WeTextProcessing (Apache-2.0 License, WeNet Team): https://github.com/wenet-e2e/WeTextProcessing
 * - PaddleSpeech Text Frontend (Apache-2.0 License, PaddlePaddle): https://github.com/PaddlePaddle/PaddleSpeech
 */
object SpeechTextFormatter {

    private const val TAG = "HelloNetraOCR"
    private const val MIN_CONFIDENCE = 0.55f
    private const val MAX_SPEECH_CHARS = 600

    fun format(results: List<OCRResult>): String {
        if (results.isEmpty()) {
            Log.d(TAG, "RAW OCR:\n[empty]")
            return ""
        }

        // 1. Log RAW OCR
        val rawOcrLog = results.joinToString(" | ") {
            "[${it.text} (conf:${String.format(Locale.US, "%.2f", it.confidence)})]"
        }
        Log.d(TAG, "RAW OCR:\n$rawOcrLog")

        // 2. Filter low-confidence OCR lines
        val confidenceFiltered = results.filter { it.confidence >= MIN_CONFIDENCE }
        val confFilteredLog = if (confidenceFiltered.isEmpty()) "[none passed]" else confidenceFiltered.joinToString("\n") { it.text }
        Log.d(TAG, "AFTER CONFIDENCE FILTER:\n$confFilteredLog")

        if (confidenceFiltered.isEmpty()) return ""

        // 3. Clean & normalize lines using WeTextProcessing / PaddleSpeech English TN rules
        val cleanedLines = confidenceFiltered
            .mapNotNull { cleanLine(it.text) }
            .filter { isMeaningful(it) }
            .map { normalizeText(it) }

        if (cleanedLines.isEmpty()) {
            Log.d(TAG, "AFTER NORMALIZATION:\n[none valid]")
            Log.d(TAG, "FINAL PIPER TEXT:\n")
            return ""
        }

        // 4. Deduplicate consecutive lines
        val deduplicated = cleanedLines.fold(mutableListOf<String>()) { acc, line ->
            if (acc.lastOrNull()?.equals(line, ignoreCase = true) != true) {
                acc.add(line)
            }
            acc
        }

        // 5. Ensure each line ends with punctuation for natural Piper pause pacing
        val formattedLines = deduplicated.map { line ->
            if (line.endsWith('.') || line.endsWith('!') || line.endsWith('?')) {
                line
            } else {
                "$line."
            }
        }

        var speech = formattedLines.joinToString(" ")
        speech = postProcessPunctuationAndSpacing(speech)

        if (speech.length > MAX_SPEECH_CHARS) {
            speech = speech.take(MAX_SPEECH_CHARS)
            val lastSpace = speech.lastIndexOf(' ')
            if (lastSpace > 0) {
                speech = speech.substring(0, lastSpace)
            }
        }

        val finalText = speech.trim()

        Log.d(TAG, "AFTER NORMALIZATION:\n${deduplicated.joinToString("\n")}")
        Log.d(TAG, "FINAL PIPER TEXT:\n$finalText")

        return finalText
    }

    private fun cleanLine(input: String): String? {
        if (input.isBlank()) return null

        var text = input

        // Filter out ISO control characters except newline and tab
        text = text.filter { !it.isISOControl() || it == '\n' || it == '\t' }

        // Normalize Unicode smart quotes, dashes, ellipsis
        text = text
            .replace('“', '"')
            .replace('”', '"')
            .replace('‘', '\'')
            .replace('’', '\'')
            .replace('–', '-')
            .replace('—', '-')
            .replace("…", "...")

        // Collapse duplicate punctuation (e.g. ..... -> ., !!! -> !)
        text = text.replace(Regex("([!?.,;:])\\1+"), "$1")

        // Collapse excessive whitespace
        text = text.replace(Regex("\\s+"), " ")

        // Trim noisy surrounding symbols
        text = text.trim(' ', '|', '•', '·', '_', '~')

        return text.ifBlank { null }
    }

    private fun isMeaningful(text: String): Boolean {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return false

        val letterOrDigitCount = trimmed.count { it.isLetterOrDigit() }

        // Single valid letter or digit (e.g., "A", "I", "1") is valid
        if (trimmed.length == 1 && letterOrDigitCount == 1) return true

        // Reject strings with 0 letters/digits (e.g. "@@@###|||")
        if (letterOrDigitCount == 0) return false

        // Require at least 20% alphanumeric content for multi-character strings
        val ratio = letterOrDigitCount.toFloat() / trimmed.length
        return ratio >= 0.20f
    }

    /**
     * Normalizes English text for speech synthesis based on WeTextProcessing / PaddleSpeech rules.
     */
    private fun normalizeText(input: String): String {
        var text = input.trim()

        // 1. Currency Symbols (e.g., ₹500 -> 500 rupees, $50 -> 50 dollars)
        text = text.replace(Regex("₹\\s*(\\d+(?:\\.\\d+)?)"), "$1 rupees")
            .replace(Regex("Rs\\.?\\s*(\\d+(?:\\.\\d+)?)", RegexOption.IGNORE_CASE), "$1 rupees")
            .replace(Regex("\\$\\s*(\\d+(?:\\.\\d+)?)"), "$1 dollars")
            .replace(Regex("£\\s*(\\d+(?:\\.\\d+)?)"), "$1 pounds")
            .replace(Regex("€\\s*(\\d+(?:\\.\\d+)?)"), "$1 euros")

        // 2. Common English Abbreviations (WeTextProcessing English TN rules)
        text = text
            .replace(Regex("\\be\\.g\\.\\s*|\\be\\.g\\b", RegexOption.IGNORE_CASE), "for example ")
            .replace(Regex("\\bi\\.e\\.\\s*|\\bi\\.e\\b", RegexOption.IGNORE_CASE), "that is ")
            .replace(Regex("\\bDr\\.\\s*", RegexOption.IGNORE_CASE), "Doctor ")
            .replace(Regex("\\bMr\\.\\s*", RegexOption.IGNORE_CASE), "Mister ")
            .replace(Regex("\\bMrs\\.\\s*", RegexOption.IGNORE_CASE), "Missus ")
            .replace(Regex("\\bMs\\.\\s*", RegexOption.IGNORE_CASE), "Miss ")
            .replace(Regex("\\bProf\\.\\s*", RegexOption.IGNORE_CASE), "Professor ")
            .replace(Regex("\\bSt\\.\\s*", RegexOption.IGNORE_CASE), "Saint ")
            .replace(Regex("\\betc\\.\\s*|\\betc\\b", RegexOption.IGNORE_CASE), "et cetera ")
            .replace(Regex("\\bvs\\.\\s*|\\bvs\\b", RegexOption.IGNORE_CASE), "versus ")
            .replace(Regex("\\bNo\\.\\s*(?=\\d)", RegexOption.IGNORE_CASE), "number ")

        // 3. Symbols
        text = text
            .replace("%", " percent")
            .replace("&", " and ")
            .replace("@", " at ")
            .replace("#", " number ")

        // 4. Time format: 12:30 -> 12 30
        text = text.replace(Regex("(\\b\\d{1,2}):(\\d{2}\\b)"), "$1 $2")

        // 5. Decimal numbers: 10.5 -> 10 point 5
        text = text.replace(Regex("(\\b\\d+)\\.(\\d+\\b)"), "$1 point $2")

        // 6. Ordinals: 1st -> first, 2nd -> second, 3rd -> third, etc.
        text = text
            .replace(Regex("\\b1st\\b", RegexOption.IGNORE_CASE), "first")
            .replace(Regex("\\b2nd\\b", RegexOption.IGNORE_CASE), "second")
            .replace(Regex("\\b3rd\\b", RegexOption.IGNORE_CASE), "third")
            .replace(Regex("\\b4th\\b", RegexOption.IGNORE_CASE), "fourth")
            .replace(Regex("\\b5th\\b", RegexOption.IGNORE_CASE), "fifth")
            .replace(Regex("\\b6th\\b", RegexOption.IGNORE_CASE), "sixth")
            .replace(Regex("\\b7th\\b", RegexOption.IGNORE_CASE), "seventh")
            .replace(Regex("\\b8th\\b", RegexOption.IGNORE_CASE), "eighth")
            .replace(Regex("\\b9th\\b", RegexOption.IGNORE_CASE), "ninth")
            .replace(Regex("\\b10th\\b", RegexOption.IGNORE_CASE), "tenth")

        // 7. Case Normalization for ALL-CAPS words (> 2 chars) to prevent unnatural spelling out
        text = normalizeWordCases(text)

        return text
    }

    /**
     * Converts ALL-CAPS words (> 2 characters) to Title Case (e.g. HELLO -> Hello, NETRA -> Netra),
     * while preserving short acronyms (e.g. OK, A, US, UK, ID, SOS, OCR, GPS, TV, PC).
     */
    private fun normalizeWordCases(input: String): String {
        val words = input.split(Regex("(?<=\\s)|(?=\\s)|(?<=[.,!?;:])|(?=[.,!?;:])"))
        return words.joinToString("") { word ->
            if (word.length > 2 && word.all { it.isUpperCase() }) {
                word.lowercase().replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.US) else it.toString() }
            } else {
                word
            }
        }
    }

    private fun postProcessPunctuationAndSpacing(input: String): String {
        var text = input

        // Remove noise dividers
        text = text.replace(Regex("\\s*[-|_•·]+\\s*"), " ")

        // Collapse multiple spaces
        text = text.replace(Regex("\\s+"), " ")

        // Remove spaces before punctuation marks
        text = text.replace(Regex("\\s+([.,!?;:])"), "$1")

        // Ensure space after punctuation marks if followed by a letter
        text = text.replace(Regex("([.!?])([A-Za-z])"), "$1 $2")

        // Capitalize sentence starts
        text = text.replace(Regex("(^[a-z]|[.!?]\\s+[a-z])")) { match ->
            match.value.uppercase(Locale.US)
        }

        return text.trim()
    }
}