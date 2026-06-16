package com.example.gpbmerclient.ui.main

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation3.runtime.NavKey
import com.example.gpbmerclient.Login
import com.example.gpbmerclient.Settings
import com.example.gpbmerclient.data.AnalysisResponse
import com.example.gpbmerclient.data.AppSettings
import com.example.gpbmerclient.data.CallHistoryItem

@Composable
fun MainScreen(
    onItemClick: (NavKey) -> Unit,
    onLogout: () -> Unit,
    modifier: Modifier = Modifier,
    viewModel: MainScreenViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    val duration by viewModel.recordingDuration.collectAsState()
    val history by viewModel.history.collectAsState()
    val offlineNotes by viewModel.offlineNotes.collectAsState()
    val isOffline by AppSettings.isOffline.collectAsState()
    
    val context = LocalContext.current
    var selectedTab by remember { mutableStateOf(0) } // 0 - Анализ, 1 - История

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { isGranted ->
            if (isGranted) {
                viewModel.startRecording()
            }
        }
    )

    // Обновляем историю при переходе на вкладку
    LaunchedEffect(selectedTab) {
        if (selectedTab == 1) {
            viewModel.refreshHistory()
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF0A0F1A))
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Шапка экрана: Пользователь, Настройки и Выход
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "🎙 GPB MER Client",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF60A5FA)
                )
                Text(
                    text = "Пользователь: ${AppSettings.username} (${if (AppSettings.role == "admin") "Админ" else "Оператор"})",
                    fontSize = 11.sp,
                    color = Color(0xFF9CA3AF)
                )
                Text(
                    text = if (isOffline) "Режим: ОФФЛАЙН 📵" else "Режим: ОНЛАЙН 📡",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (isOffline) Color(0xFFF59E0B) else Color(0xFF10B981)
                )
            }
            
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = { onItemClick(Settings) }) {
                    Icon(
                        imageVector = Icons.Default.Settings,
                        contentDescription = "Настройки",
                        tint = Color.White
                    )
                }
                Spacer(modifier = Modifier.width(4.dp))
                Button(
                    onClick = {
                        AppSettings.username = ""
                        AppSettings.role = "user"
                        onLogout()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF7F1D1D)),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                ) {
                    Text("Выйти", color = Color.White, fontSize = 12.sp)
                }
            }
        }

        // Переключатель вкладок
        TabRow(
            selectedTabIndex = selectedTab,
            containerColor = Color(0xFF1E293B),
            contentColor = Color(0xFF60A5FA),
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp)
        ) {
            Tab(
                selected = selectedTab == 0,
                onClick = { selectedTab = 0 },
                text = { Text("Анализатор", color = if (selectedTab == 0) Color.White else Color(0xFF9CA3AF)) }
            )
            Tab(
                selected = selectedTab == 1,
                onClick = { selectedTab = 1 },
                text = { Text("История звонков", color = if (selectedTab == 1) Color.White else Color(0xFF9CA3AF)) }
            )
        }

        if (selectedTab == 0) {
            // ВКЛАДКА АНАЛИЗАТОРА
            Column(
                modifier = Modifier
                    .weight(1f)
                    .verticalScroll(rememberScrollState()),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Если мы оффлайн, показываем поле ввода заметок оператора
                if (isOffline) {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 16.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(
                                text = "📝 Оффлайн-заметки диалога",
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                color = Color.White,
                                modifier = Modifier.padding(bottom = 8.dp)
                            )
                            OutlinedTextField(
                                value = offlineNotes,
                                onValueChange = { viewModel.updateOfflineNotes(it) },
                                placeholder = { Text("Введите важные моменты диалога...") },
                                modifier = Modifier.fillMaxWidth().height(80.dp),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedTextColor = Color.White,
                                    unfocusedTextColor = Color.White,
                                    focusedBorderColor = Color(0xFF3B82F6),
                                    unfocusedBorderColor = Color(0xFF475569)
                                )
                            )
                        }
                    }
                }

                // Блок управления записью
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 20.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFF111827)),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        if (uiState is ClientUiState.Recording) {
                            Text(
                                text = "Идёт запись аудио...",
                                fontSize = 16.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = Color(0xFFEF4444)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = String.format("%02d:%02d", duration / 60, duration % 60),
                                fontSize = 32.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White
                            )
                            Spacer(modifier = Modifier.height(20.dp))
                            Button(
                                onClick = { viewModel.stopAndAnalyze() },
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEF4444)),
                                shape = CircleShape,
                                modifier = Modifier.size(80.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(24.dp)
                                        .background(Color.White, RoundedCornerShape(4.dp))
                                )
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = if (isOffline) "Нажмите для локального анализа" else "Нажмите для отправки по VPN",
                                fontSize = 12.sp,
                                color = Color(0xFF9CA3AF)
                            )
                        } else {
                            Text(
                                text = "Готов к записи",
                                fontSize = 16.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = Color(0xFF10B981)
                            )
                            Spacer(modifier = Modifier.height(20.dp))
                            Button(
                                onClick = {
                                    val hasPermission = ContextCompat.checkSelfPermission(
                                        context,
                                        Manifest.permission.RECORD_AUDIO
                                    ) == PackageManager.PERMISSION_GRANTED
                                    if (hasPermission) {
                                        viewModel.startRecording()
                                    } else {
                                        permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                    }
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF3B82F6)),
                                shape = CircleShape,
                                modifier = Modifier.size(80.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.PlayArrow,
                                    contentDescription = "Start",
                                    tint = Color.White,
                                    modifier = Modifier.size(36.dp)
                                )
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = "Нажмите, чтобы начать запись",
                                fontSize = 12.sp,
                                color = Color(0xFF9CA3AF)
                            )
                        }
                    }
                }

                // Отображение результатов анализа
                when (val state = uiState) {
                    is ClientUiState.Loading -> {
                        CircularProgressIndicator(color = Color(0xFF3B82F6))
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = if (isOffline) "Локальная обработка..." else "Отправка и ИИ-анализ...",
                            color = Color(0xFF9CA3AF),
                            fontSize = 14.sp
                        )
                    }
                    is ClientUiState.Error -> {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFF7F1D1D)),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Text(
                                    text = "⚠️ Ошибка выполнения",
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFFFCA5A5),
                                    fontSize = 16.sp,
                                    modifier = Modifier.padding(bottom = 4.dp)
                                )
                                Text(
                                    text = state.message,
                                    color = Color(0xFFFECACA),
                                    fontSize = 14.sp
                                )
                            }
                        }
                    }
                    is ClientUiState.Success -> {
                        AnalysisResultView(response = state.response)
                    }
                    else -> {}
                }
            }
        } else {
            // ВКЛАДКА ИСТОРИИ ЗВОНКОВ
            Column(
                modifier = Modifier
                    .weight(1f)
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = if (AppSettings.role == "admin") "📖 Общая история (Админ)" else "📖 Моя история звонков",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    modifier = Modifier.padding(bottom = 12.dp)
                )

                if (history.isEmpty()) {
                    Text(
                        text = "История звонков пуста.",
                        color = Color(0xFF9CA3AF),
                        fontSize = 14.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth().padding(top = 40.dp)
                    )
                } else {
                    history.forEach { item ->
                        HistoryCard(item = item)
                        Spacer(modifier = Modifier.height(12.dp))
                    }
                }
            }
        }
    }
}

@Composable
fun AnalysisResultView(response: AnalysisResponse) {
    val stressPercent = (response.final_stress * 100).toInt()
    
    val cardColor = when {
        stressPercent >= 70 -> Color(0xFF7F1D1D)
        stressPercent >= 40 -> Color(0xFF78350F)
        else -> Color(0xFF064E3B)
    }

    val statusText = when {
        stressPercent >= 70 -> "Критический стресс / Аномалия"
        stressPercent >= 40 -> "Повышенное волнение"
        else -> "Нормальное / Стабильное состояние"
    }

    Column(modifier = Modifier.fillMaxWidth()) {
        // Карточка индекса стресса
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            colors = CardDefaults.cardColors(containerColor = cardColor),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "ИНДЕКС АНОМАЛИИ / СТРЕССА",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White.copy(alpha = 0.7f)
                )
                Text(
                    text = "$stressPercent%",
                    fontSize = 44.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = Color.White
                )
                Text(
                    text = statusText,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color.White
                )
            }
        }

        // Блок конспектирования (Саммари)
        if (response.summary.isNotBlank()) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "📌 Конспект / Саммари диалога:",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF60A5FA),
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Text(
                        text = response.summary,
                        fontSize = 14.sp,
                        color = Color.White,
                        lineHeight = 22.sp
                    )
                }
            }
        }

        // Карточка транскрипции
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "📝 Распознанный текст (ASR):",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF9CA3AF),
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                Text(
                    text = "\"${response.transcription}\"",
                    fontSize = 15.sp,
                    fontStyle = FontStyle.Italic,
                    color = Color.White,
                    lineHeight = 22.sp
                )
            }
        }

        // Блок проверок регламентов (QA)
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "📊 Проверка регламента (QA):",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF9CA3AF),
                    modifier = Modifier.padding(bottom = 12.dp)
                )
                
                ComplianceItem(label = "Приветствие", isPassed = response.compliance.greeting)
                Spacer(modifier = Modifier.height(8.dp))
                ComplianceItem(label = "Прощание", isPassed = response.compliance.goodbye)
                Spacer(modifier = Modifier.height(8.dp))
                ComplianceItem(label = "Вежливость", isPassed = response.compliance.politeness)
                Spacer(modifier = Modifier.height(8.dp))
                ComplianceItem(label = "Отсутствие стоп-слов", isPassed = response.compliance.no_stop_words)
            }
        }

        // Блок акустических характеристик
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "🎙 Акустические характеристики:",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF9CA3AF),
                    modifier = Modifier.padding(bottom = 12.dp)
                )
                
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    MetricText(label = "Длительность", value = "${response.features.duration} с")
                    MetricText(label = "Темп речи", value = "${response.features.tempo_bpm} BPM")
                }
                Spacer(modifier = Modifier.height(8.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    MetricText(label = "Громкость", value = String.format("%.1f", response.features.loudness_mean))
                    MetricText(label = "Паузы/тишина", value = "${(response.features.silence_ratio * 100).toInt()}%")
                }
            }
        }
    }
}

@Composable
fun ComplianceItem(label: String, isPassed: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = if (isPassed) Icons.Default.Check else Icons.Default.Close,
            contentDescription = null,
            tint = if (isPassed) Color(0xFF10B981) else Color(0xFFEF4444),
            modifier = Modifier.size(20.dp)
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = label,
            fontSize = 14.sp,
            color = Color.White
        )
    }
}

@Composable
fun MetricText(label: String, value: String) {
    Column {
        Text(text = label, fontSize = 12.sp, color = Color(0xFF9CA3AF))
        Text(text = value, fontSize = 16.sp, fontWeight = FontWeight.Bold, color = Color.White)
    }
}

@Composable
fun HistoryCard(item: CallHistoryItem) {
    var expanded by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { expanded = !expanded },
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Сотрудник: ${item.operator}",
                        fontWeight = FontWeight.Bold,
                        color = Color.White,
                        fontSize = 14.sp
                    )
                    Text(
                        text = item.timestamp,
                        color = Color(0xFF9CA3AF),
                        fontSize = 12.sp
                    )
                }
                
                Column(horizontalAlignment = Alignment.End) {
                    Text(
                        text = "Стресс: ${(item.stress_score * 100).toInt()}%",
                        color = if (item.stress_score >= 0.4) Color(0xFFEF4444) else Color(0xFF10B981),
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp
                    )
                    Text(
                        text = "Длительность: ${item.duration.toInt()} с",
                        color = Color(0xFF9CA3AF),
                        fontSize = 11.sp
                    )
                }
            }
            
            if (expanded) {
                Divider(
                    color = Color(0xFF475569),
                    thickness = 1.dp,
                    modifier = Modifier.padding(vertical = 12.dp)
                )
                
                Text(
                    text = "Конспект разговора:",
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF60A5FA),
                    fontSize = 13.sp,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
                
                Text(
                    text = item.summary,
                    color = Color.White,
                    fontSize = 13.sp,
                    lineHeight = 20.sp
                )
                
                Spacer(modifier = Modifier.height(10.dp))
                
                Text(
                    text = "Транскрипция:",
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF9CA3AF),
                    fontSize = 13.sp,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
                
                Text(
                    text = "\"${item.transcription}\"",
                    color = Color.White,
                    fontStyle = FontStyle.Italic,
                    fontSize = 13.sp
                )
            }
        }
    }
}
