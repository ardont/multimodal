package com.example.gpbmerclient

import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.runtime.rememberNavBackStack
import androidx.navigation3.ui.NavDisplay
import com.example.gpbmerclient.ui.login.LoginScreen
import com.example.gpbmerclient.ui.main.MainScreen
import com.example.gpbmerclient.ui.settings.SettingsScreen

@Composable
fun MainNavigation() {
  val backStack = rememberNavBackStack(Login)

  NavDisplay(
    backStack = backStack,
    onBack = { backStack.removeLastOrNull() },
    entryProvider =
      entryProvider {
        entry<Login> {
          LoginScreen(
            onLoginSuccess = { backStack.add(Main) },
            modifier = Modifier.safeDrawingPadding().padding(16.dp)
          )
        }
        entry<Main> {
          MainScreen(
            onItemClick = { navKey -> backStack.add(navKey) },
            onLogout = { backStack.add(Login) },
            modifier = Modifier.safeDrawingPadding().padding(16.dp)
          )
        }
        entry<Settings> {
          SettingsScreen(
            onBackClick = { backStack.removeLastOrNull() },
            modifier = Modifier.safeDrawingPadding().padding(16.dp)
          )
        }
      },
  )
}
