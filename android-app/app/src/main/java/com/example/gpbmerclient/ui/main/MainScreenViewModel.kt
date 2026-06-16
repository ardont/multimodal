package com.example.gpbmerclient.ui.main

import android.app.Application
import android.os.Handler
import android.os.Looper
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.gpbmerclient.data.AcousticFeatures
import com.example.gpbmerclient.data.AnalysisClient
import com.example.gpbmerclient.data.AnalysisResponse
import com.example.gpbmerclient.data.AppSettings
import com.example.gpbmerclient.data.AudioRecorder
import com.example.gpbmerclient.data.CallHistoryItem
import com.example.gpbmerclient.data.ComplianceResponse
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File

sealed interface ClientUiState {
    object Idle : ClientUiState
    object Recording : ClientUiState
    object Loading : ClientUiState
    data class Success(val response: AnalysisResponse) : ClientUiState
    data class Error(val message: String) : ClientUiState
}

class MainScreenViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow<ClientUiState>(ClientUiState.Idle)
    val uiState: StateFlow<ClientUiState> = _uiState.asStateFlow()

    private val _recordingDuration = MutableStateFlow(0)
    val recordingDuration: StateFlow<Int> = _recordingDuration.asStateFlow()

    private val _history = MutableStateFlow<List<CallHistoryItem>>(emptyList())
    val history: StateFlow<List<CallHistoryItem>> = _history.asStateFlow()

    private val _offlineNotes = MutableStateFlow("")
    val offlineNotes: StateFlow<String> = _offlineNotes.asStateFlow()

    private val context = application.applicationContext
    private var audioFile: File = File(context.cacheDir, "record.wav")
    private var recorder: AudioRecorder? = null
    private val client = AnalysisClient()

    private var durationHandler: Handler? = null
    private val durationRunnable = object : Runnable {
        override fun run() {
            _recordingDuration.update { it + 1 }
            durationHandler?.postDelayed(this, 1000)
        }
    }

    init {
        // Подгружаем историю звонков при запуске
        refreshHistory()
    }

    fun updateOfflineNotes(notes: String) {
        _offlineNotes.value = notes
    }

    fun refreshHistory() {
        if (AppSettings.isOffline.value) {
            // Оффлайн-история остается локальной (то, что сохранили в памяти)
            return
        }
        viewModelScope.launch {
            client.fetchHistory(
                AppSettings.serverAddress.value,
                AppSettings.username,
                AppSettings.role
            ) { items, err ->
                if (items != null) {
                    _history.value = items
                }
            }
        }
    }

    fun startRecording() {
        try {
            audioFile.delete()
            _recordingDuration.value = 0
            _uiState.value = ClientUiState.Recording
            
            recorder = AudioRecorder(audioFile)
            recorder?.startRecording()

            durationHandler = Handler(Looper.getMainLooper())
            durationHandler?.postDelayed(durationRunnable, 1000)
        } catch (e: Exception) {
            _uiState.value = ClientUiState.Error("Не удалось начать запись: ${e.message}")
        }
    }

    fun stopAndAnalyze() {
        if (_uiState.value != ClientUiState.Recording) return
        
        try {
            durationHandler?.removeCallbacks(durationRunnable)
            durationHandler = null
            
            recorder?.stopRecording()
            val rec = recorder
            recorder = null

            _uiState.value = ClientUiState.Loading

            val isOffline = AppSettings.isOffline.value
            val durationSec = _recordingDuration.value

            if (isOffline) {
                // Оффлайн анализ на основе локального расчета
                val rms = rec?.lastRmsDb ?: -50.0
                val silence = rec?.lastSilenceRatio ?: 0.1
                
                // Перевод децибел в относительную "громкость" от 0 до 100
                val loudness = ((rms + 100).coerceIn(0.0, 100.0)) / 10.0
                
                // Эмулируем локальную логику
                val text = "Разговор записан локально оффлайн."
                val textNotes = _offlineNotes.value
                
                val summary = """
                    📊 **Локальный оффлайн-анализ диалога**
                    🔊 Громкость: ${String.format("%.1f", loudness)} (RMS: ${String.format("%.1f", rms)} dB)
                    ⏱ Длительность: $durationSec с
                    🔇 Доля тишины: ${(silence * 100).toInt()}%
                    
                    📝 **Заметки оператора:**
                    ${if (textNotes.isNotBlank()) textNotes else "Нет пользовательских заметок."}
                    
                    ⚠️ *Локальный режим: голосовой ИИ (ASR) недоступен без сети.*
                """.trimIndent()

                val localResponse = AnalysisResponse(
                    transcription = text,
                    text_stress = 0.0f,
                    audio_stress = 0.0f,
                    final_stress = 0.0f,
                    features = AcousticFeatures(
                        duration = durationSec.toDouble(),
                        loudness_mean = loudness,
                        silence_ratio = silence,
                        tempo_bpm = 120.0
                    ),
                    compliance = ComplianceResponse(
                        greeting = true,
                        goodbye = true,
                        politeness = true,
                        no_stop_words = true
                    ),
                    summary = summary
                )

                _uiState.value = ClientUiState.Success(localResponse)
                
                // Сохраняем звонок в локальную историю в памяти
                val compScore = 1.0
                val localItem = CallHistoryItem(
                    operator = AppSettings.username,
                    timestamp = "Только что (оффлайн)",
                    duration = durationSec.toDouble(),
                    stress_score = 0.0,
                    compliance_score = compScore,
                    summary = summary,
                    transcription = text
                )
                _history.value = listOf(localItem) + _history.value

            } else {
                // Онлайн анализ через VPN
                client.analyzeAudio(AppSettings.serverAddress.value, audioFile, AppSettings.username) { response ->
                    if (response.error != null) {
                        _uiState.value = ClientUiState.Error(response.error)
                    } else {
                        _uiState.value = ClientUiState.Success(response)
                        refreshHistory()
                    }
                }
            }
        } catch (e: Exception) {
            _uiState.value = ClientUiState.Error("Ошибка анализа: ${e.message}")
        }
    }

    override fun onCleared() {
        super.onCleared()
        durationHandler?.removeCallbacks(durationRunnable)
    }
}
