package com.example.gpbmerclient.ui.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.gpbmerclient.data.AppSettings

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBackClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val serverAddressFlow = AppSettings.serverAddress.collectAsState()
    var serverAddress by remember { mutableStateOf(serverAddressFlow.value) }

    val isOfflineFlow = AppSettings.isOffline.collectAsState()
    val isOffline = isOfflineFlow.value

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Настройки", color = Color.White) },
                navigationIcon = {
                    TextButton(onClick = onBackClick) {
                        Text("← Назад", color = Color(0xFF60A5FA), fontSize = 16.sp)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = Color(0xFF1E293B))
            )
        },
        containerColor = Color(0xFF0A0F1A)
    ) { innerPadding ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp)
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "⚙ Параметры работы",
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp,
                        color = Color.White,
                        modifier = Modifier.padding(bottom = 16.dp)
                    )

                    OutlinedTextField(
                        value = serverAddress,
                        onValueChange = {
                            serverAddress = it
                            AppSettings.setServerAddress(it)
                        },
                        label = { Text("Адрес сервера VPN") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White,
                            focusedBorderColor = Color(0xFF3B82F6),
                            unfocusedBorderColor = Color(0xFF475569)
                        )
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text(
                                text = "Оффлайн режим",
                                fontWeight = FontWeight.Bold,
                                color = Color.White,
                                fontSize = 16.sp
                            )
                            Text(
                                text = "Анализ будет производиться без интернета локально на телефоне",
                                color = Color(0xFF9CA3AF),
                                fontSize = 11.sp,
                                modifier = Modifier.fillMaxWidth(0.75f)
                            )
                        }

                        Switch(
                            checked = isOffline,
                            onCheckedChange = { AppSettings.setOffline(it) },
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = Color(0xFF3B82F6),
                                checkedTrackColor = Color(0xFF1E3A8A)
                            )
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(20.dp))

            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "👤 Текущий пользователь",
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp,
                        color = Color.White,
                        modifier = Modifier.padding(bottom = 12.dp)
                    )
                    Text(
                        text = "Имя: ${AppSettings.username}",
                        color = Color.White,
                        fontSize = 14.sp
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Роль: ${if (AppSettings.role == "admin") "Администратор" else "Оператор"}",
                        color = Color(0xFF9CA3AF),
                        fontSize = 14.sp
                    )
                }
            }
        }
    }
}
