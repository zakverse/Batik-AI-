/// Centralized API configuration for Wastra AI Batik.
///
/// For development:
/// - Android Emulator : 'http://10.0.2.2:8080'
/// - iOS Simulator    : 'http://localhost:8080'
/// - Physical Device  : 'http://<YOUR-LOCAL-IP>:8080' (e.g. 'http://192.168.1.100:8080')
class ApiConfig {
  // Default to Android Emulator address (can be updated for physical device or iOS)
  static const String baseUrl = 'http://10.0.2.2:8080';

  // API Endpoints
  static const String predictEndpoint = '$baseUrl/api/v1/predict';
  static const String healthEndpoint = '$baseUrl/health';

  // Timeout settings
  static const Duration requestTimeout = Duration(seconds: 30);

  // Helper to build Predict URI with query parameters
  static Uri predictUri({int topK = 3}) {
    return Uri.parse('$predictEndpoint?top_k=$topK');
  }

  static Uri healthUri() {
    return Uri.parse(healthEndpoint);
  }
}
