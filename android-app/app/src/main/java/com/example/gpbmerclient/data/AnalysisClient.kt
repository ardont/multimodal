package com.example.gpbmerclient.data

import com.google.gson.Gson
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.util.concurrent.TimeUnit

data class AcousticFeatures(
    val duration: Double = 0.0,
    val loudness_mean: Double = 0.0,
    val silence_ratio: Double = 0.0,
    val tempo_bpm: Double = 0.0
)

data class ComplianceResponse(
    val greeting: Boolean = false,
    val goodbye: Boolean = false,
    val politeness: Boolean = false,
    val no_stop_words: Boolean = false,
    val found_stops: List<String> = emptyList()
)

data class AnalysisResponse(
    val transcription: String = "",
    val text_stress: Float = 0.0f,
    val audio_stress: Float = 0.0f,
    val final_stress: Float = 0.0f,
    val features: AcousticFeatures = AcousticFeatures(),
    val compliance: ComplianceResponse = ComplianceResponse(),
    val summary: String = "",
    val error: String? = null
)

data class LoginRequestBody(val username: String, val password: String)

data class LoginResponse(
    val success: Boolean,
    val role: String? = null,
    val username: String? = null,
    val error: String? = null
)

data class CallHistoryItem(
    val operator: String = "",
    val timestamp: String = "",
    val duration: Double = 0.0,
    val stress_score: Double = 0.0,
    val compliance_score: Double = 0.0,
    val summary: String = "",
    val transcription: String = ""
)

class AnalysisClient {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val gson = Gson()

    fun analyzeAudio(serverAddress: String, audioFile: File, operator: String?, callback: (AnalysisResponse) -> Unit) {
        Thread {
            try {
                val url = if (serverAddress.startsWith("http://") || serverAddress.startsWith("https://")) {
                    "$serverAddress/api/analyze"
                } else {
                    "http://$serverAddress/api/analyze"
                }

                val mediaType = "audio/wav".toMediaType()
                val requestBodyBuilder = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart(
                        "file",
                        audioFile.name,
                        audioFile.asRequestBody(mediaType)
                    )
                if (operator != null) {
                    requestBodyBuilder.addFormDataPart("operator", operator)
                }
                val requestBody = requestBodyBuilder.build()

                val request = Request.Builder()
                    .url(url)
                    .post(requestBody)
                    .build()

                client.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        val bodyString = response.body?.string() ?: ""
                        val result = gson.fromJson(bodyString, AnalysisResponse::class.java)
                        callback(result)
                    } else {
                        callback(AnalysisResponse(error = "HTTP Error: ${response.code}"))
                    }
                }
            } catch (e: Exception) {
                callback(AnalysisResponse(error = "Connection failed: ${e.message}"))
            }
        }.start()
    }

    fun performLogin(serverAddress: String, requestBody: LoginRequestBody, callback: (LoginResponse) -> Unit) {
        Thread {
            try {
                val url = if (serverAddress.startsWith("http://") || serverAddress.startsWith("https://")) {
                    "$serverAddress/api/login"
                } else {
                    "http://$serverAddress/api/login"
                }

                val mediaType = "application/json".toMediaType()
                val jsonBody = gson.toJson(requestBody)
                val body = jsonBody.toRequestBody(mediaType)

                val request = Request.Builder()
                    .url(url)
                    .post(body)
                    .build()

                client.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        val bodyString = response.body?.string() ?: ""
                        val result = gson.fromJson(bodyString, LoginResponse::class.java)
                        callback(result)
                    } else {
                        callback(LoginResponse(success = false, error = "HTTP Error: ${response.code}"))
                    }
                }
            } catch (e: Exception) {
                callback(LoginResponse(success = false, error = "Connection failed: ${e.message}"))
            }
        }.start()
    }

    fun fetchHistory(serverAddress: String, username: String, role: String, callback: (List<CallHistoryItem>?, String?) -> Unit) {
        Thread {
            try {
                val url = if (serverAddress.startsWith("http://") || serverAddress.startsWith("https://")) {
                    "$serverAddress/api/history?username=$username&role=$role"
                } else {
                    "http://$serverAddress/api/history?username=$username&role=$role"
                }

                val request = Request.Builder()
                    .url(url)
                    .get()
                    .build()

                client.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        val bodyString = response.body?.string() ?: ""
                        val items = gson.fromJson(bodyString, Array<CallHistoryItem>::class.java).toList()
                        callback(items, null)
                    } else {
                        callback(null, "HTTP Error: ${response.code}")
                    }
                }
            } catch (e: Exception) {
                callback(null, "Connection failed: ${e.message}")
            }
        }.start()
    }
}
