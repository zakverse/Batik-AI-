/// Centralized API configuration for NusantaraKain.
///
/// For development:
/// - Android Emulator : 'http://10.0.2.2:8080'
/// - iOS Simulator    : 'http://localhost:8080'
/// - Physical Device  : 'http://<YOUR-LOCAL-IP>:8080' (e.g. 'http://192.168.1.100:8080')
class ApiConfig {
  // Base URL for API service (http://127.0.0.1:8080 works for Android Physical Device via ADB reverse, iOS Simulator, and Desktop)
  static const String baseUrl = 'http://127.0.0.1:8080';

  // API Endpoints (36-Class Modern V2 API)
  static const String predictEndpoint = '$baseUrl/api/v2/predict';
  static const String healthEndpoint = '$baseUrl/api/v2/health';

  // Timeout settings
  static const Duration requestTimeout = Duration(seconds: 30);

  // Helper to build Predict URI with query parameters (top_k between 1 and 36)
  static Uri predictUri({int topK = 3}) {
    return Uri.parse('$predictEndpoint?top_k=$topK');
  }

  static Uri healthUri() {
    return Uri.parse(healthEndpoint);
  }
}
