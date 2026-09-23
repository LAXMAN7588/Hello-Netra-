package com.example.objectdetection

/**
 * Operating modes for the Hello-Netra system.
 *
 * Controlled automatically by physical hardware buttons via ESP32-C3 -> Raspberry Pi -> Android:
 * - IDLE: Standby state, server ready, heavy ONNX vision models paused to save battery/RAM.
 * - OBJECT_RECOGNITION: Real-time YOLOv8 object detection with >=60% confidence filter and Piper TTS.
 * - OCR: PP-OCRv5 text recognition pipeline with CTC decoding and Piper TTS speech.
 * - SOS: Emergency location acquisition and automated direct SMS transmission to all saved contacts.
 */
enum class AppMode {
    IDLE,
    OBJECT_RECOGNITION,
    OCR,
    SOS
}
