#include <jni.h>
#include <string>
#include <vector>
#include <espeak-ng/speak_lib.h>
#include <android/log.h>

#define LOG_TAG "HelloNetraPhonemizer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static bool espeak_initialized = false;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_hellonetra_ocr_NativePhonemizer_initializeEspeak(JNIEnv *env, jobject thiz, jstring data_path) {
    if (espeak_initialized) return JNI_TRUE;
    
    const char *path = env->GetStringUTFChars(data_path, nullptr);
    LOGI("Initializing eSpeak-NG with data path: %s", path);
    
    // AUDIO_OUTPUT_RETRIEVAL: no audio playback, just text-to-phonemes
    int sample_rate = espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0, path, 0);
    env->ReleaseStringUTFChars(data_path, path);
    
    if (sample_rate < 0) {
        LOGE("Failed to initialize eSpeak-NG (returned %d)", sample_rate);
        return JNI_FALSE;
    }
    
    // Set voice to en-us (matches espeak.voice in the Indian Piper model JSON)
    if (espeak_SetVoiceByName("en-us") != EE_OK) {
        LOGE("Failed to set eSpeak-NG voice to en-us");
        return JNI_FALSE;
    }
    
    espeak_initialized = true;
    LOGI("eSpeak-NG initialized successfully (sample rate: %d)", sample_rate);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_hellonetra_ocr_NativePhonemizer_textToPhonemes(JNIEnv *env, jobject thiz, jstring text) {
    if (!espeak_initialized) {
        LOGE("Cannot convert to phonemes: eSpeak-NG not initialized");
        return env->NewStringUTF("");
    }
    
    const char *c_text = env->GetStringUTFChars(text, nullptr);
    
    // espeak_TextToPhonemes processes one clause at a time.
    // We must loop until textptr becomes NULL (end of input).
    std::string result;
    const void *text_ptr = c_text;
    
    while (text_ptr != nullptr) {
        // textmode = espeakCHARS_UTF8 (1)
        // phonememode = espeakPHONEMES_IPA (2)
        const char *phonemes = espeak_TextToPhonemes(&text_ptr, 1, 2);
        
        if (phonemes != nullptr && phonemes[0] != '\0') {
            if (!result.empty()) {
                // Add a space between clauses
                result += ' ';
            }
            result += phonemes;
        }
    }
    
    LOGI("textToPhonemes: IPA result length=%zu", result.size());
    
    env->ReleaseStringUTFChars(text, c_text);
    
    return env->NewStringUTF(result.c_str());
}
