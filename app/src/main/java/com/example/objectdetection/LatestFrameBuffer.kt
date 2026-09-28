package com.example.objectdetection

import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Thread-safe single-slot buffer implementing a strict "latest-frame" strategy.
 * If new frames arrive while the consumer is still processing, stale frames are dropped
 * immediately to maintain lowest real-time latency.
 */
class LatestFrameBuffer {

    private val lock = ReentrantLock()
    private val condition = lock.newCondition()

    private var latestPacket: FramePacket? = null

    private val receivedCounter = AtomicLong(0)
    private val processedCounter = AtomicLong(0)
    private val droppedCounter = AtomicLong(0)

    val framesReceived: Long get() = receivedCounter.get()
    val framesProcessed: Long get() = processedCounter.get()
    val framesDropped: Long get() = droppedCounter.get()

    /**
     * Enqueues a newly received frame packet, overwriting any waiting frame.
     */
    fun offer(packet: FramePacket) {
        receivedCounter.incrementAndGet()

        lock.withLock {
            if (latestPacket != null) {
                // Previous frame was not consumed in time -> drop it
                droppedCounter.incrementAndGet()
            }
            latestPacket = packet
            condition.signal()
        }
    }

    /**
     * Blocks until a new frame is available, then returns it.
     */
    @Throws(InterruptedException::class)
    fun take(): FramePacket {
        lock.withLock {
            while (latestPacket == null) {
                condition.await()
            }
            val packet = latestPacket!!
            latestPacket = null
            processedCounter.incrementAndGet()
            return packet
        }
    }

    /**
     * Polls the latest frame if one is immediately available without blocking.
     */
    fun poll(): FramePacket? {
        lock.withLock {
            val packet = latestPacket ?: return null
            latestPacket = null
            processedCounter.incrementAndGet()
            return packet
        }
    }

    fun clear() {
        lock.withLock {
            latestPacket = null
        }
    }

    fun resetCounters() {
        receivedCounter.set(0)
        processedCounter.set(0)
        droppedCounter.set(0)
        clear()
    }
}
