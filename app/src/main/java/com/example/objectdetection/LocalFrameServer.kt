package com.example.objectdetection

import android.util.Log
import java.io.InputStream
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Local TCP Server running on the Android phone to receive video frames from Raspberry Pi.
 */
class LocalFrameServer(
    val port: Int = 5000,
    private val onConnectionStateChanged: (ConnectionState) -> Unit,
    private val onFrameReceived: (FramePacket) -> Unit
) {
    companion object {
        private const val TAG = "LocalFrameServer"
        private const val READ_BUFFER_SIZE = 16 * 1024
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
            server.bind(InetSocketAddress(port))
            serverSocket = server

            val localIp = IpUtils.getLocalIpAddress()
            Log.i(TAG, "Local Frame Server listening on $localIp:$port")
            onConnectionStateChanged(ConnectionState.Listening(port, localIp))

            while (isRunning.get() && !server.isClosed) {
                try {
                    Log.i(TAG, "Waiting for Raspberry Pi connection...")
                    val client = server.accept()
                    clientSocket = client
                    client.tcpNoDelay = true
                    client.receiveBufferSize = 64 * 1024

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

            parser.feedData(buffer, bytesRead)
        }
    }

    private fun closeClientSocket() {
        try {
            clientSocket?.close()
        } catch (_: Exception) {
        }
        clientSocket = null
        parser.reset()
    }

    fun stop() {
        isRunning.set(false)
        closeClientSocket()
        try {
            serverSocket?.close()
        } catch (_: Exception) {
        }
        serverSocket = null
        onConnectionStateChanged(ConnectionState.Stopped)
    }

    fun release() {
        stop()
        serverExecutor.shutdownNow()
    }
}
