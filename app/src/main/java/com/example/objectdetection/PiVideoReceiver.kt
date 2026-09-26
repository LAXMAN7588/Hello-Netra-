
package com.example.objectdetection

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaFormat
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import java.io.ByteArrayOutputStream
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetSocketAddress
import java.nio.ByteBuffer
import java.util.TreeMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean


class PiVideoReceiver(
    private val width: Int = 640,
    private val height: Int = 480,
    private val port: Int = 5000,
    private val onVideoFrame: (Bitmap) -> Unit,
    private val onStatus: (String) -> Unit,
    private val onStats: (Int, Double) -> Unit
) {

    companion object {

        private const val TAG = "PiVideoReceiver"

        /*
         * Pi packet format:
         *
         * 4 bytes  MAGIC = NETR
         * 4 bytes  FRAME ID
         * 2 bytes  PACKET ID
         * 2 bytes  PACKET COUNT
         * 4 bytes  FRAME SIZE
         *
         * Total = 16 bytes
         */
        private const val HEADER_SIZE = 16

        private const val MAX_PACKET_SIZE = 1500

        private const val MAGIC_0 = 'N'.code.toByte()
        private const val MAGIC_1 = 'E'.code.toByte()
        private const val MAGIC_2 = 'T'.code.toByte()
        private const val MAGIC_3 = 'R'.code.toByte()

        private const val FRAME_TIMEOUT_MS = 250L
    }


    // ========================================================
    // FRAME BUFFER
    // ========================================================

    private data class FrameBuffer(
        val frameId: Int,
        val packetCount: Int,
        val frameSize: Int,
        val packets: Array<ByteArray?>,
        var received: Int = 0,
        val createdAt: Long =
            SystemClock.elapsedRealtime()
    )


    // ========================================================
    // STATE
    // ========================================================

    private val running =
        AtomicBoolean(false)

    private var socket: DatagramSocket? = null

    private var receiveThread: Thread? = null

    /*
     * Decoder runs independently from UDP reception.
     */
    private val decoderExecutor =
        Executors.newSingleThreadExecutor()

    private var decoder: MediaCodec? = null

    /*
     * H264 decoder output Surface.
     */
    private var decoderSurface: Surface? = null

    /*
     * ImageReader receives decoded YUV frames.
     */
    private var imageReader: ImageReader? = null

    private var imageReaderThread: HandlerThread? = null

    private var imageReaderHandler: Handler? = null


    // ========================================================
    // FRAME REASSEMBLY
    // ========================================================

    private val frames =
        TreeMap<Int, FrameBuffer>()

    private val frameLock =
        Any()

    private var lastFrameId = -1


    // ========================================================
    // STATS
    // ========================================================

    private var frameCounter = 0

    private var fpsStartTime =
        SystemClock.elapsedRealtime()

    private var fpsFrameCount = 0


    // ========================================================
    // START
    // ========================================================

    fun start() {

        if (running.get()) {
            Log.d(TAG, "Already running")
            return
        }

        running.set(true)

        frameCounter = 0
        fpsFrameCount = 0
        fpsStartTime =
            SystemClock.elapsedRealtime()

        lastFrameId = -1

        synchronized(frameLock) {
            frames.clear()
        }

        onStatus("Starting Pi video receiver...")

        /*
         * Create decoder first.
         */
        decoderExecutor.execute {

            createDecoder()

            if (!running.get()) {
                return@execute
            }

            if (decoder == null) {

                onStatus(
                    "H264 decoder could not start"
                )

                return@execute
            }

            onStatus(
                "H264 decoder ready"
            )
        }

        /*
         * Start UDP receiver.
         */
        receiveThread =
            Thread {

                receiveLoop()

            }.apply {

                name = "PiUDPReceiver"

            }

        receiveThread?.start()
    }


    // ========================================================
    // CREATE IMAGE READER + H264 DECODER
    // ========================================================

    private fun createDecoder() {

        try {

            Log.i(
                TAG,
                "Creating ImageReader ${width}x${height}"
            )

            /*
             * ImageReader receives the decoded YUV frames.
             *
             * YUV_420_888 is preferred over NV21 because
             * MediaCodec may use different YUV plane layouts.
             */
            imageReader =
                ImageReader.newInstance(
                    width,
                    height,
                    ImageFormat.YUV_420_888,
                    3
                )

            /*
             * Dedicated thread for decoded images.
             */
            imageReaderThread =
                HandlerThread(
                    "PiDecodedImageThread"
                )

            imageReaderThread?.start()

            imageReaderHandler =
                Handler(
                    imageReaderThread!!.looper
                )

            imageReader?.setOnImageAvailableListener(
                { reader ->

                    processDecodedImage(reader)

                },
                imageReaderHandler
            )

            decoderSurface =
                imageReader!!.surface

            /*
             * Create H264 decoder.
             */
            val format =
                MediaFormat.createVideoFormat(
                    MediaFormat.MIMETYPE_VIDEO_AVC,
                    width,
                    height
                )

            format.setInteger(
                MediaFormat.KEY_MAX_INPUT_SIZE,
                2 * 1024 * 1024
            )

            /*
             * Some devices need the color format to remain
             * decoder-selected when using a Surface.
             */
            Log.i(
                TAG,
                "Creating H264 MediaCodec..."
            )

            decoder =
                MediaCodec.createDecoderByType(
                    MediaFormat.MIMETYPE_VIDEO_AVC
                )

            decoder!!.configure(
                format,
                decoderSurface,
                null,
                0
            )

            decoder!!.start()

            Log.i(
                TAG,
                "H264 MediaCodec started"
            )

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Decoder creation failed",
                e
            )

            onStatus(
                "H264 decoder error: ${e.message}"
            )

            releaseDecoder()
        }
    }


    // ========================================================
    // UDP RECEIVE LOOP
    // ========================================================

    private fun receiveLoop() {

        try {

            /*
             * Create socket.
             */
            socket =
                DatagramSocket(null)

            socket?.reuseAddress = true

            socket?.receiveBufferSize =
                4 * 1024 * 1024

            socket?.bind(
                InetSocketAddress(port)
            )

            onStatus(
                "UDP listening on port $port"
            )

            Log.i(
                TAG,
                "UDP socket listening on $port"
            )

            val buffer =
                ByteArray(MAX_PACKET_SIZE)

            while (running.get()) {

                val packet =
                    DatagramPacket(
                        buffer,
                        buffer.size
                    )

                try {

                    socket?.receive(packet)

                } catch (e: Exception) {

                    if (running.get()) {

                        Log.e(
                            TAG,
                            "UDP receive error",
                            e
                        )
                    }

                    break
                }

                if (!running.get()) {
                    break
                }

                if (packet.length < HEADER_SIZE) {
                    continue
                }

                processPacket(
                    packet.data,
                    packet.length
                )

                cleanupOldFrames()
            }

        } catch (e: Exception) {

            if (running.get()) {

                Log.e(
                    TAG,
                    "UDP receiver failure",
                    e
                )

                onStatus(
                    "UDP error: ${e.message}"
                )
            }

        } finally {

            try {
                socket?.close()
            } catch (_: Exception) {
            }

            socket = null
        }
    }


    // ========================================================
    // PROCESS UDP PACKET
    // ========================================================

    private fun processPacket(
        data: ByteArray,
        length: Int
    ) {

        if (length < HEADER_SIZE) {
            return
        }

        /*
         * Check NETR magic.
         */
        if (
            data[0] != MAGIC_0 ||
            data[1] != MAGIC_1 ||
            data[2] != MAGIC_2 ||
            data[3] != MAGIC_3
        ) {
            return
        }

        /*
         * Header:
         *
         * 0-3   MAGIC
         * 4-7   FRAME ID
         * 8-9   PACKET ID
         * 10-11 PACKET COUNT
         * 12-15 FRAME SIZE
         */

        val frameId =
            readInt(
                data,
                4
            )

        val packetId =
            readUnsignedShort(
                data,
                8
            )

        val packetCount =
            readUnsignedShort(
                data,
                10
            )

        val frameSize =
            readInt(
                data,
                12
            )

        /*
         * Validate.
         */
        if (packetCount <= 0) {
            return
        }

        if (packetId >= packetCount) {
            return
        }

        if (frameSize <= 0) {
            return
        }

        val payloadSize =
            length - HEADER_SIZE

        if (payloadSize <= 0) {
            return
        }

        /*
         * Copy payload because DatagramPacket's buffer
         * is reused.
         */
        val payload =
            ByteArray(payloadSize)

        System.arraycopy(
            data,
            HEADER_SIZE,
            payload,
            0,
            payloadSize
        )

        synchronized(frameLock) {

            /*
             * Ignore old frames.
             */
            if (
                lastFrameId >= 0 &&
                frameId < lastFrameId
            ) {
                return
            }

            /*
             * Create frame buffer.
             */
            var frame =
                frames[frameId]

            if (frame == null) {

                frame =
                    FrameBuffer(
                        frameId = frameId,
                        packetCount = packetCount,
                        frameSize = frameSize,
                        packets =
                            arrayOfNulls(
                                packetCount
                            )
                    )

                frames[frameId] =
                    frame
            }

            /*
             * Safety check in case packet count changes.
             */
            if (
                frame.packetCount !=
                packetCount
            ) {
                frames.remove(frameId)
                return
            }

            /*
             * Ignore duplicate packet.
             */
            if (
                frame.packets[packetId] == null
            ) {

                frame.packets[packetId] =
                    payload

                frame.received++
            }

            /*
             * Complete H264 chunk/frame.
             */
            if (
                frame.received ==
                frame.packetCount
            ) {

                val complete =
                    rebuildFrame(frame)

                frames.remove(frameId)

                lastFrameId =
                    maxOf(
                        lastFrameId,
                        frameId
                    )

                if (complete != null) {

                    /*
                     * Feed decoder on decoder thread.
                     */
                    decoderExecutor.execute {

                        feedDecoder(
                            complete,
                            frameId
                        )
                    }
                }
            }

            /*
             * Keep low latency.
             */
            while (frames.size > 4) {

                frames.remove(
                    frames.firstKey()
                )
            }
        }
    }


    // ========================================================
    // REBUILD H264 DATA
    // ========================================================

    private fun rebuildFrame(
        frame: FrameBuffer
    ): ByteArray? {

        /*
         * Verify every packet exists.
         */
        for (packet in frame.packets) {

            if (packet == null) {
                return null
            }
        }

        val output =
            ByteArrayOutputStream(
                frame.frameSize
            )

        for (packet in frame.packets) {

            if (packet == null) {
                return null
            }

            output.write(packet)
        }

        val result =
            output.toByteArray()

        /*
         * Verify size against Pi's advertised frame size.
         */
        if (result.size != frame.frameSize) {

            Log.w(
                TAG,
                "Frame size mismatch: " +
                        "expected=${frame.frameSize}, " +
                        "actual=${result.size}"
            )

            return null
        }

        return result
    }


    // ========================================================
    // FEED H264 DECODER
    // ========================================================

    private fun feedDecoder(
        h264Data: ByteArray,
        frameId: Int
    ) {

        if (!running.get()) {
            return
        }

        val codec =
            decoder ?: return

        try {

            /*
             * H264 access unit input.
             */
            val inputIndex =
                codec.dequeueInputBuffer(
                    10_000
                )

            if (inputIndex < 0) {

                Log.w(
                    TAG,
                    "No decoder input buffer"
                )

                return
            }

            val inputBuffer =
                codec.getInputBuffer(
                    inputIndex
                )

            if (inputBuffer == null) {
                return
            }

            inputBuffer.clear()

            if (
                h264Data.size >
                inputBuffer.remaining()
            ) {

                Log.e(
                    TAG,
                    "H264 frame too large: " +
                            "${h264Data.size} bytes"
                )

                return
            }

            inputBuffer.put(
                h264Data
            )

            /*
             * 20 FPS => 50 ms per frame.
             */
            val presentationTimeUs =
                frameId.toLong() *
                        50_000L

            codec.queueInputBuffer(
                inputIndex,
                0,
                h264Data.size,
                presentationTimeUs,
                0
            )

            /*
             * Because decoder output goes to Surface,
             * release output buffers with render=true.
             */
            drainDecoder(codec)

        } catch (e: Exception) {

            if (running.get()) {

                Log.e(
                    TAG,
                    "H264 decoder input error",
                    e
                )
            }
        }
    }


    // ========================================================
    // DRAIN DECODER
    // ========================================================

    private fun drainDecoder(
        codec: MediaCodec
    ) {

        val info =
            MediaCodec.BufferInfo()

        while (running.get()) {

            val outputIndex =
                codec.dequeueOutputBuffer(
                    info,
                    0
                )

            when {

                outputIndex ==
                        MediaCodec.INFO_TRY_AGAIN_LATER -> {

                    break
                }

                outputIndex ==
                        MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {

                    Log.i(
                        TAG,
                        "Decoder format changed: " +
                                codec.outputFormat
                    )
                }

                outputIndex >= 0 -> {

                    /*
                     * IMPORTANT:
                     *
                     * Because decoder output is connected to
                     * ImageReader Surface, render=true sends
                     * the decoded frame to ImageReader.
                     */
                    try {

                        codec.releaseOutputBuffer(
                            outputIndex,
                            true
                        )

                    } catch (e: Exception) {

                        Log.e(
                            TAG,
                            "releaseOutputBuffer error",
                            e
                        )
                    }
                }
            }
        }
    }


    // ========================================================
    // IMAGE READER
    // ========================================================

    private fun processDecodedImage(
        reader: ImageReader
    ) {

        if (!running.get()) {
            return
        }

        var image: Image? = null

        try {

            /*
             * acquireLatestImage() intentionally drops old
             * frames if the detector/display is slower.
             *
             * This keeps the system low latency.
             */
            image =
                reader.acquireLatestImage()

            if (image == null) {
                return
            }

            val bitmap =
                imageToBitmap(image)

            if (bitmap != null) {

                handleDecodedFrame(
                    bitmap
                )
            }

        } catch (e: Exception) {

            if (running.get()) {

                Log.e(
                    TAG,
                    "Decoded image error",
                    e
                )
            }

        } finally {

            try {
                image?.close()
            } catch (_: Exception) {
            }
        }
    }


    // ========================================================
    // IMAGE -> BITMAP
    // ========================================================

    private fun imageToBitmap(
        image: Image
    ): Bitmap? {

        return try {

            val planes =
                image.planes

            if (planes.size < 3) {
                return null
            }

            val width =
                image.width

            val height =
                image.height

            /*
             * YUV_420_888 can have row/pixel strides that
             * differ between devices.
             *
             * Convert it to NV21 manually.
             */
            val nv21 =
                ByteArray(
                    width * height * 3 / 2
                )

            val yPlane =
                planes[0]

            val uPlane =
                planes[1]

            val vPlane =
                planes[2]

            val yBuffer =
                yPlane.buffer

            val uBuffer =
                uPlane.buffer

            val vBuffer =
                vPlane.buffer

            val yRowStride =
                yPlane.rowStride

            val yPixelStride =
                yPlane.pixelStride

            val uRowStride =
                uPlane.rowStride

            val uPixelStride =
                uPlane.pixelStride

            val vRowStride =
                vPlane.rowStride

            val vPixelStride =
                vPlane.pixelStride

            /*
             * Y
             */
            var outputOffset = 0

            val yRow =
                ByteArray(
                    yRowStride
                )

            for (row in 0 until height) {

                val rowStart =
                    row * yRowStride

                val rowLength =
                    minOf(
                        yRowStride,
                        yBuffer.remaining()
                    )

                yBuffer.position(
                    minOf(
                        rowStart,
                        yBuffer.limit()
                    )
                )

                val available =
                    minOf(
                        rowLength,
                        yBuffer.remaining()
                    )

                if (available > 0) {

                    yBuffer.get(
                        yRow,
                        0,
                        available
                    )
                }

                for (col in 0 until width) {

                    val sourceIndex =
                        col * yPixelStride

                    if (
                        sourceIndex <
                        available
                    ) {

                        nv21[
                            outputOffset++
                        ] =
                            yRow[sourceIndex]
                    }
                }
            }

            /*
             * VU
             *
             * NV21 expects V then U.
             */
            val chromaHeight =
                height / 2

            val chromaWidth =
                width / 2

            val uRow =
                ByteArray(
                    uRowStride
                )

            val vRow =
                ByteArray(
                    vRowStride
                )

            for (row in 0 until chromaHeight) {

                val uRowStart =
                    row * uRowStride

                val vRowStart =
                    row * vRowStride

                uBuffer.position(
                    minOf(
                        uRowStart,
                        uBuffer.limit()
                    )
                )

                vBuffer.position(
                    minOf(
                        vRowStart,
                        vBuffer.limit()
                    )
                )

                val uAvailable =
                    minOf(
                        uRowStride,
                        uBuffer.remaining()
                    )

                val vAvailable =
                    minOf(
                        vRowStride,
                        vBuffer.remaining()
                    )

                if (uAvailable > 0) {

                    uBuffer.get(
                        uRow,
                        0,
                        uAvailable
                    )
                }

                if (vAvailable > 0) {

                    vBuffer.get(
                        vRow,
                        0,
                        vAvailable
                    )
                }

                for (col in 0 until chromaWidth) {

                    val uIndex =
                        col * uPixelStride

                    val vIndex =
                        col * vPixelStride

                    if (
                        vIndex < vAvailable &&
                        uIndex < uAvailable
                    ) {

                        nv21[
                            outputOffset++
                        ] =
                            vRow[vIndex]

                        nv21[
                            outputOffset++
                        ] =
                            uRow[uIndex]
                    }
                }
            }

            /*
             * NV21 -> Bitmap
             */
            val yuvImage =
                android.graphics.YuvImage(
                    nv21,
                    ImageFormat.NV21,
                    width,
                    height,
                    null
                )

            val jpeg =
                ByteArrayOutputStream()

            yuvImage.compressToJpeg(
                android.graphics.Rect(
                    0,
                    0,
                    width,
                    height
                ),
                85,
                jpeg
            )

            Bitmap.createBitmap(
                BitmapFactoryCompat.decode(
                    jpeg.toByteArray()
                )
            )

        } catch (e: Exception) {

            Log.e(
                TAG,
                "YUV -> Bitmap failed",
                e
            )

            null
        }
    }


    // ========================================================
    // DECODE JPEG
    // ========================================================

    private object BitmapFactoryCompat {

        fun decode(
            data: ByteArray
        ): Bitmap {

            return android.graphics.BitmapFactory
                .decodeByteArray(
                    data,
                    0,
                    data.size
                )
                ?: throw IllegalStateException(
                    "Unable to decode JPEG"
                )
        }
    }


    // ========================================================
    // DECODED FRAME HANDLER
    // ========================================================

    private fun handleDecodedFrame(
        bitmap: Bitmap
    ) {

        if (!running.get()) {

            bitmap.recycle()

            return
        }

        frameCounter++
        fpsFrameCount++

        val now =
            SystemClock.elapsedRealtime()

        val elapsed =
            now - fpsStartTime

        if (elapsed >= 1000L) {

            val fps =
                fpsFrameCount *
                        1000.0 /
                        elapsed.toDouble()

            fpsFrameCount = 0
            fpsStartTime = now

            onStats(
                frameCounter,
                fps
            )
        }

        /*
         * Deliver bitmap to MainActivity.
         */
        onVideoFrame(bitmap)
    }


    // ========================================================
    // REMOVE OLD INCOMPLETE FRAMES
    // ========================================================

    private fun cleanupOldFrames() {

        val now =
            SystemClock.elapsedRealtime()

        synchronized(frameLock) {

            val iterator =
                frames.entries.iterator()

            while (iterator.hasNext()) {

                val entry =
                    iterator.next()

                val frame =
                    entry.value

                if (
                    now - frame.createdAt >
                    FRAME_TIMEOUT_MS
                ) {

                    iterator.remove()
                }
            }
        }
    }


    // ========================================================
    // STOP
    // ========================================================

    fun stop() {

        if (!running.getAndSet(false)) {
            return
        }

        onStatus(
            "Stopping Pi video..."
        )

        Log.i(
            TAG,
            "Stopping PiVideoReceiver"
        )

        /*
         * Stop UDP.
         */
        try {
            socket?.close()
        } catch (_: Exception) {
        }

        socket = null

        /*
         * Clear incomplete frames.
         */
        synchronized(frameLock) {
            frames.clear()
        }

        /*
         * Stop decoder.
         */
        decoderExecutor.execute {

            try {
                decoder?.stop()
            } catch (_: Exception) {
            }

            try {
                decoder?.release()
            } catch (_: Exception) {
            }

            decoder = null

            try {
                imageReader?.close()
            } catch (_: Exception) {
            }

            imageReader = null

            try {
                decoderSurface?.release()
            } catch (_: Exception) {
            }

            decoderSurface = null

            try {
                imageReaderThread?.quitSafely()
            } catch (_: Exception) {
            }

            imageReaderThread = null
            imageReaderHandler = null
        }

        /*
         * Stop UDP thread.
         */
        try {
            receiveThread?.interrupt()
        } catch (_: Exception) {
        }

        receiveThread = null

        onStatus(
            "Pi video stopped"
        )
    }


    // ========================================================
    // RELEASE DECODER
    // ========================================================

    private fun releaseDecoder() {

        try {
            decoder?.stop()
        } catch (_: Exception) {
        }

        try {
            decoder?.release()
        } catch (_: Exception) {
        }

        decoder = null

        try {
            imageReader?.close()
        } catch (_: Exception) {
        }

        imageReader = null

        try {
            decoderSurface?.release()
        } catch (_: Exception) {
        }

        decoderSurface = null

        try {
            imageReaderThread?.quitSafely()
        } catch (_: Exception) {
        }

        imageReaderThread = null
        imageReaderHandler = null
    }


    // ========================================================
    // HELPERS
    // ========================================================

    private fun readUnsignedShort(
        data: ByteArray,
        offset: Int
    ): Int {

        return (
                ((data[offset].toInt() and 0xFF) shl 8)
                        or
                        (data[offset + 1].toInt() and 0xFF)
                )
    }


    private fun readInt(
        data: ByteArray,
        offset: Int
    ): Int {

        return (
                ((data[offset].toInt() and 0xFF) shl 24)
                        or
                        ((data[offset + 1].toInt() and 0xFF) shl 16)
                        or
                        ((data[offset + 2].toInt() and 0xFF) shl 8)
                        or
                        (data[offset + 3].toInt() and 0xFF)
                )
    }
}

