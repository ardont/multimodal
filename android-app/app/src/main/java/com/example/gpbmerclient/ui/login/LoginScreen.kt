package com.example.gpbmerclient.ui.login

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.gpbmerclient.data.AnalysisClient
import com.example.gpbmerclient.data.AppSettings
import com.example.gpbmerclient.data.LoginRequestBody

@Composable
fun LoginScreen(
    onLoginSuccess: () -> Unit,
    modifier: Modifier = Modifier
) {
    var username by remember { mutableStateOf("operator") }
    var password by remember { mutableStateOf("operator") }
    
    val serverAddressFlow = AppSettings.serverAddress.collectAsState()
    var serverAddress by remember { mutableStateOf(serverAddressFlow.value) }
    
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    val isOfflineFlow = AppSettings.isOffline.collectAsState()
    val isOffline = isOfflineFlow.value

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF0A0F1A))
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Логотип и заголовок
        Text(
            text = "🎙 GPB MER",
            fontSize = 36.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF60A5FA),
            textAlign = TextAlign.Center
        )
        Text(
            text = "Анализ речи & Комплаенс",
            fontSize = 16.sp,
            color = Color(0xFF9CA3AF),
            modifier = Modifier.padding(top = 4.dp, bottom = 40.dp),
            textAlign = TextAlign.Center
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Авторизация",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    modifier = Modifier.padding(bottom = 20.dp)
                )

                // Поле ввода сервера
                OutlinedTextField(
                    value = serverAddress,
                    onValueChange = {
                        serverAddress = it
                        AppSettings.setServerAddress(it)
                    },
                    label = { Text("Адрес сервера VPN") },
                    placeholder = { Text("100.90.91.54:7860") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedBorderColor = Color(0xFF3B82F6),
                        unfocusedBorderColor = Color(0xFF475569)
                    )
                )
                Spacer(modifier = Modifier.height(12.dp))

                // Логин
                OutlinedTextField(
                    value = username,
                    onValueChange = { username = it },
                    label = { Text("Имя пользователя") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedBorderColor = Color(0xFF3B82F6),
                        unfocusedBorderColor = Color(0xFF475569)
                    )
                )
                Spacer(modifier = Modifier.height(12.dp))

                // Пароль
                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text("Пароль") },
                    visualTransformation = PasswordVisualTransformation(),
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedBorderColor = Color(0xFF3B82F6),
                        unfocusedBorderColor = Color(0xFF475569)
                    )
                )
                Spacer(modifier = Modifier.height(24.dp))

                // Кнопка переключения режима
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Оффлайн режим (без сети)",
                        fontSize = 12.sp,
                        color = Color(0xFF9CA3AF)
                    )
                    Switch(
                        checked = isOffline,
                        onCheckedChange = { AppSettings.setOffline(it) },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color(0xFF3B82F6),
                            checkedTrackColor = Color(0xFF1E3A8A)
                        )
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))

                if (errorMessage != null) {
                    Text(
                        text = errorMessage!!,
                        color = Color(0xFFEF4444),
                        fontSize = 14.sp,
                        modifier = Modifier.padding(bottom = 12.dp),
                        textAlign = TextAlign.Center
                    )
                }

                if (isLoading) {
                    CircularProgressIndicator(color = Color(0xFF3B82F6))
                } else {
                    Button(
                        onClick = {
                            if (isOffline) {
                                AppSettings.username = username
                                AppSettings.role = if (username == "admin") "admin" else "user"
                                onLoginSuccess()
                            } else {
                                isLoading = true
                                errorMessage = null
                                val client = AnalysisClient()
                                client.performLogin(serverAddress, LoginRequestBody(username, password)) { res ->
                                    isLoading = false
                                    if (res.success) {
                                        AppSettings.username = res.username ?: username
                                        AppSettings.role = res.role ?: "user"
                                        onLoginSuccess()
                                    } else {
                                        errorMessage = res.error ?: "Ошибка подключения"
                                    }
                                }
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF3B82F6)),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Войти", color = Color.White, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}
