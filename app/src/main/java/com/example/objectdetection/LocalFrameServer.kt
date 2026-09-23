package com.example.objectdetection

import android.util.Log
import com.example.objectdetection.safety.TofSensorManager
import java.io.InputStream
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Local TCP Server running on the Android phone to receive:
 * 1. Control commands (MODE_OBJECT, MODE_OCR, MODE_STOP, MODE_SOS)
 * 2. Binary Video Frame Packets (Magic "PIFR")
 * 3. Binary ToF Safety Packets (Magic "PITF" - 45 bytes) / Text "TOF:" packets
 *
 * Binds to 0.0.0.0 (wildcard interface) to accept incoming connections across all hotspot subnets.
 */
class LocalFrameServer(
    val port: Int = 5000,
    private val onConnectionStateChanged: (ConnectionState) -> Unit,
    private val onControlCommandReceived: (AppMode) -> Unit,
    private val onFrameReceived: (FramePacket) -> Unit,
    private val onTofBinaryReceived: ((ByteArray, Int, Int) -> Unit)? = null,
    private val onTofTextReceived: ((String) -> Unit)? = null
) {
    companion object {
        private const val TAG = "LocalFrameServer"
        private const val READ_BUFFER_SIZE = 32 * 1024
    }

    sealed class ConnectionState {
        object Stopped : ConnectionState()
        data class Listening(val port: Int, val localIp: String) : ConnectionState()
        data class Connected(val clientAddress: String) : ConnectionState()
        data class Disconnected(val reason: String) : ConnectionState()
        data class Error(val message: String) : ConnectionState()
    }

    private val isRunning = AtomicBoolean(false)
    private var serverSocket: ServerSocket? = null
    private var clientSocket: Socket? = null
    private val serverExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private val parser = FramePacketParser { packet ->
        onFrameReceived(packet)
    }

    fun start() {
        if (isRunning.getAndSet(true)) {
            Log.w(TAG, "Server already running.")
            return
        }

        serverExecutor.execute {
            runServerLoop()
        }
    }

    private fun runServerLoop() {
        try {
            val server = ServerSocket()
            server.reuseAddress = true
            server.bind(InetSocketAddress("0.0.0.0", port))
            serverSocket = server

            val localIp = IpUtils.getLocalIpAddress()
            Log.i(TAG, "Local Frame Server listening on 0.0.0.0:$port (Detected IP: $localIp)")
            onConnectionStateChanged(ConnectionState.Listening(port, localIp))

            while (isRunning.get() && !server.isClosed) {
                try {
                    Log.i(TAG, "Waiting for Raspberry Pi connection...")
                    val client = server.accept()
                    clientSocket = client
                    client.tcpNoDelay = true
                    client.receiveBufferSize = 128 * 1024

                    val clientAddr = client.remoteSocketAddress.toString()
                    Log.i(TAG, "Raspberry Pi connected from $clientAddr")
                    onConnectionStateChanged(ConnectionState.Connected(clientAddr))

                    parser.reset()
                    handleClient(client)

                } catch (e: Exception) {
                    if (isRunning.get() && !server.isClosed) {
                        Log.i(TAG, "Client disconnected: ${e.message}")
                        onConnectionStateChanged(ConnectionState.Disconnected(e.message ?: "Disconnected"))
                    }
                } finally {
                    closeClientSocket()
                }
            }
        } catch (e: Exception) {
            if (isRunning.get()) {
                Log.e(TAG, "Server error on port $port", e)
                onConnectionStateChanged(ConnectionState.Error("Server error: ${e.message}"))
            }
        } finally {
            stop()
        }
    }

    private fun handleClient(client: Socket) {
        val inputStream: InputStream = client.getInputStream()
        val buffer = ByteArray(READ_BUFFER_SIZE)

        while (isRunning.get() && !client.isClosed) {
            val bytesRead = inputStream.read(buffer)
            if (bytesRead == -1) {
                Log.i(TAG, "End of stream reached from client.")
                break
            }

            var offset = 0
            while (offset < bytesRead) {
                val remaining = bytesRead - offset

                // 1. Check for Binary ToF Packet: Magic "PITF" (45 bytes)
                if (remaining >= TofSensorManager.BINARY_PACKET_SIZE &&
                    buffer[offset] == TofSensorManager.MAGIC_BYTES[0] &&
                    buffer[offset + 1] == TofSensorManager.MAGIC_BYTES[1] &&
                    buffer[offset + 2] == TofSensorManager.MAGIC_BYTES[2] &&
                    buffer[offset + 3] == TofSensorManager.MAGIC_BYTES[3]
                ) {
                    onTofBinaryReceived?.invoke(buffer, offset, TofSensorManager.BINARY_PACKET_SIZE)
                    offset += TofSensorManager.BINARY_PACKET_SIZE
                    continue
                }

                // 2. Check for Text Messages (Commands or "TOF:" telemetry)
                if (isPotentialControlMessage(buffer, offset, remaining)) {
                    val lineEnd = findLineEnd(buffer, offset, remaining)
                    val len = if (lineEnd != -1) (lineEnd - offset) else remaining
                    val text = String(buffer, offset, len, Charsets.UTF_8).trim()

                    if (text.startsWith("TOF:", ignoreCase = true)) {
                        onTofTextReceived?.invoke(text)
                        offset += if (lineEnd != -1) (lineEnd - offset + 1) else remaining
                        continue
                    }

                    val mode = PiControlProtocol.parseCommand(text)
                    if (mode != null) {
                        Log.i(TAG, "Received control command from Pi: $text -> $mode")
                        onControlCommandReceived(mode)
                        offset += if (lineEnd != -1) (lineEnd - offset + 1) else remaining
                        continue
                    }
                }

                // 3. Feed video stream data to FramePacketParser
                val chunkLen = remaining
                val chunk = ByteArray(chunkLen)
                System.arraycopy(buffer, offset, chunk, 0, chunkLen)
                parser.feedData(chunk, chunkLen)
                break
            }
        }
    }

    private fun findLineEnd(data: ByteArray, startOffset: Int, length: Int): Int {
        for (i in startOffset until (startOffset + length)) {
            if (data[i] == '\n'.code.toByte() || data[i] == '\r'.code.toByte()) {
                return i
            }
        }
        return -1
    }

    private fun isPotentialControlMessage(data: ByteArray, startOffset: Int, length: Int): Boolean {
        if (length < 3) return false

        // Exclude binary video header "PIFR"
        if (length >= 4 &&
            data[startOffset] == 'P'.code.toByte() &&
            data[startOffset + 1] == 'I'.code.toByte() &&
            data[startOffset + 2] == 'F'.code.toByte() &&
            data[startOffset + 3] == 'R'.code.toByte()
        ) {
            return false
        }

        // Exclude binary ToF header "PITF"
        if (length >= 4 &&
            data[startOffset] == 'P'.code.toByte() &&
            data[startOffset + 1] == 'I'.code.toByte() &&
            data[startOffset + 2] == 'T'.code.toByte() &&
            data[startOffset + 3] == 'F'.code.toByte()
        ) {
            return false
        }

        // Check if characters are printable ASCII
        for (i in startOffset until minOf(startOffset + length, startOffset + 32)) {
            val b = data[i].toInt()
            if (b == '\n'.code || b == '\r'.code) break
            if (b < 32 || b > 126) return false
        }
        return true
    }

    private fun closeClientSocket() {
        try {
            clientSocket?.close()
        } catch (_: Exception) {}
        clientSocket = null
        parser.reset()
    }

    fun stop() {
        isRunning.set(false)
        closeClientSocket()
        try {
            serverSocket?.close()
        } catch (_: Exception) {}
        serverSocket = null
        onConnectionStateChanged(ConnectionState.Stopped)
    }

    fun release() {
        stop()
        serverExecutor.shutdownNow()
    }
}
