import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:wastra_ai_mobile/main.dart';
import 'package:wastra_ai_mobile/models/prediction_response.dart';
import 'package:wastra_ai_mobile/screens/result_screen.dart';
import 'package:wastra_ai_mobile/widgets/confidence_bar.dart';
import 'package:wastra_ai_mobile/widgets/image_source_button.dart';

void main() {
  group('App Smoke & Widget Tests', () {
    testWidgets('WastraApp renders HomeScreen with hero and buttons', (WidgetTester tester) async {
      await tester.pumpWidget(const WastraApp());
      await tester.pumpAndSettle();

      // Check AppBar Title
      expect(find.text('Wastra AI Batik'), findsOneWidget);

      // Check Hero Section Content
      expect(find.text('Kenali Motif Batik Indonesia'), findsOneWidget);
      expect(find.text('AI Vision Classifier'), findsOneWidget);

      // Check Buttons
      expect(find.text('Ambil Foto'), findsOneWidget);
      expect(find.text('Pilih dari Galeri'), findsOneWidget);
      expect(find.byType(ImageSourceButton), findsNWidgets(2));
    });

    testWidgets('About dialog opens and displays model information', (WidgetTester tester) async {
      await tester.pumpWidget(const WastraApp());
      await tester.pumpAndSettle();

      // Tap info icon button
      final infoButton = find.byTooltip('Tentang Aplikasi');
      expect(infoButton, findsOneWidget);
      await tester.tap(infoButton);
      await tester.pumpAndSettle();

      // Verify dialog contents
      expect(find.text('Tentang Wastra AI'), findsOneWidget);
      expect(find.textContaining('86.05%'), findsOneWidget);
      expect(find.textContaining('35 Kelas Motif Batik'), findsOneWidget);

      // Dismiss dialog
      await tester.tap(find.text('Tutup'));
      await tester.pumpAndSettle();
      expect(find.text('Tentang Wastra AI'), findsNothing);
    });

    testWidgets('ConfidenceBar renders with clamped percentage and colors', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: ConfidenceBar(
              confidence: 0.8998,
              showPercentage: true,
            ),
          ),
        ),
      );

      expect(find.text('89.98%'), findsOneWidget);
    });

    testWidgets('PredictionCard and ResultScreen render top prediction and top 3 possibilities', (WidgetTester tester) async {
      const dummyResponse = PredictionResponse(
        success: true,
        prediction: PredictionItem(
          className: 'batik-bali',
          confidence: 0.8998,
        ),
        topPredictions: [
          PredictionItem(className: 'batik-bali', confidence: 0.8998),
          PredictionItem(className: 'Maluku_Pala', confidence: 0.0351),
          PredictionItem(className: 'batik-keraton', confidence: 0.0151),
        ],
      );

      await tester.pumpWidget(
        const MaterialApp(
          home: ResultScreen(
            response: dummyResponse,
          ),
        ),
      );
      await tester.pumpAndSettle();

      // Verify header and motif titles
      expect(find.text('Hasil Analisis'), findsOneWidget);
      expect(find.text('MOTIF TERDETEKSI'), findsOneWidget);
      expect(find.text('BATIK BALI'), findsOneWidget);
      expect(find.text('89.98%'), findsNWidgets(2)); // Badge & Rank 1
      expect(find.text('Maluku Pala'), findsOneWidget);
      expect(find.text('3.51%'), findsOneWidget);
      expect(find.text('Batik Keraton'), findsOneWidget);
      expect(find.text('1.51%'), findsOneWidget);

      // Verify button
      expect(find.text('Analisis Motif Lain'), findsOneWidget);
    });
  });
}
