package com.example.gpbmerclient.data

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

object AppSettings {
    var username: String = "operator"
    var role: String = "user"
    
    private val _isOffline = MutableStateFlow(false)
    val isOffline: StateFlow<Boolean> = _isOffline
    
    fun setOffline(offline: Boolean) {
        _isOffline.value = offline
    }
    
    private val _serverAddress = MutableStateFlow("100.90.91.54:7860")
    val serverAddress: StateFlow<String> = _serverAddress
    
    fun setServerAddress(address: String) {
        _serverAddress.value = address
    }
}
