package com.example.gpbmerclient.data

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.io.FileOutputStream

class AudioRecorder(private val outputFile: File) {

    private var audioRecord: AudioRecord? = null
    private var isRecording = false

    var lastRmsDb: Double = 0.0
        private set
    var lastSilenceRatio: Double = 0.0
        private set

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

    @SuppressLint("MissingPermission")
    fun startRecording() {
        if (isRecording) return
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )

        audioRecord?.startRecording()
        isRecording = true

        Thread {
            writeAudioDataToFile()
        }.start()
    }

    fun stopRecording() {
        if (!isRecording) return
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    private fun writeAudioDataToFile() {
        val tempFile = File(outputFile.absolutePath + ".temp")
        val data = ByteArray(bufferSize)
        val os = FileOutputStream(tempFile)

        audioRecord?.let { recorder ->
            while (isRecording) {
                val read = recorder.read(data, 0, bufferSize)
                if (read > 0) {
                    os.write(data, 0, read)
                }
            }
        }
        os.close()

        rawToWave(tempFile, outputFile)
        tempFile.delete()
    }

    private fun rawToWave(rawFile: File, waveFile: File) {
        val rawData = rawFile.readBytes()
        analyzePcmData(rawData)
        val fileOutputStream = FileOutputStream(waveFile)
        
        val totalAudioLen = rawData.size.toLong()
        val totalDataLen = totalAudioLen + 36
        val longSampleRate = sampleRate.toLong()
        val channels = 1
        val byteRate = 16 * sampleRate * channels / 8

        val header = ByteArray(44)
        header[0] = 'R'.code.toByte() // RIFF/WAVE header
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte() // 'fmt ' chunk
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16 // 4 bytes: size of 'fmt ' chunk
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // format = 1 (PCM)
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (longSampleRate and 0xff).toByte()
        header[25] = ((longSampleRate shr 8) and 0xff).toByte()
        header[26] = ((longSampleRate shr 16) and 0xff).toByte()
        header[27] = ((longSampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = (1 * 16 / 8).toByte() // block align
        header[33] = 0
        header[34] = 16 // bits per sample
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xff).toByte()

        fileOutputStream.write(header, 0, 44)
        fileOutputStream.write(rawData)
        fileOutputStream.close()
    }

    private fun analyzePcmData(rawData: ByteArray) {
        val shorts = ShortArray(rawData.size / 2)
        var sumSquares = 0.0
        var silenceSamples = 0
        val threshold = 0.03 * 32768.0 // 3% от макс амплитуды
        
        for (i in shorts.indices) {
            val low = rawData[i * 2].toInt()
            val high = rawData[i * 2 + 1].toInt()
            val sample = ((high shl 8) or (low and 0xff)).toShort()
            shorts[i] = sample
            
            sumSquares += sample.toDouble() * sample.toDouble()
            if (Math.abs(sample.toInt()) < threshold) {
                silenceSamples++
            }
        }
        
        val rms = if (shorts.isNotEmpty()) Math.sqrt(sumSquares / shorts.size) else 0.0
        val normalizedRms = rms / 32768.0
        lastRmsDb = if (normalizedRms > 0.0) 20 * Math.log10(normalizedRms) else -100.0
        lastSilenceRatio = if (shorts.isNotEmpty()) silenceSamples.toDouble() / shorts.size else 0.0
    }
}
