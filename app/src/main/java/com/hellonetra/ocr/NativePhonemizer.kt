package com.hellonetra.ocr

object NativePhonemizer {
    init {
        System.loadLibrary("hellonetra_phonemizer")
    }

    /**
     * Initializes eSpeak-NG with the given path to espeak-ng-data.
     * The path should point to the directory containing the 'espeak-ng-data' folder,
     * or to the folder itself (usually espeak_Initialize wants the path up to, but not including, espeak-ng-data).
     * Actually, espeak_Initialize path parameter should be the directory that CONTAINS espeak-ng-data.
     */
    external fun initializeEspeak(dataPath: String): Boolean

    /**
     * Converts text to espeak phonemes.
     */
    external fun textToPhonemes(text: String): String
}
