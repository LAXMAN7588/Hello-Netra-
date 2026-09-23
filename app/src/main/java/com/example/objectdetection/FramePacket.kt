package com.example.objectdetection

/**
 * Encapsulates a complete reconstructed frame packet received over the TCP network.
 *
 * @param frameId Unique monotonically increasing identifier of the frame
 * @param timestampMs Epoch timestamp in milliseconds when the frame was captured by the Pi
 * @param format Image format code (1 = JPEG)
 * @param jpegData Raw byte array containing the JPEG image
 */
data class FramePacket(
    val frameId: Long,
    val timestampMs: Long,
    val format: Byte,
    val jpegData: ByteArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as FramePacket
        if (frameId != other.frameId) return false
        if (timestampMs != other.timestampMs) return false
        if (format != other.format) return false
        if (!jpegData.contentEquals(other.jpegData)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = frameId.hashCode()
        result = 31 * result + timestampMs.hashCode()
        result = 31 * result + format.toInt()
        result = 31 * result + jpegData.contentHashCode()
        return result
    }
}
