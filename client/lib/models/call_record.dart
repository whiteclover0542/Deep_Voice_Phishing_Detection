import 'analysis_result.dart';

class CallRecord {
  final String id;
  final String text;
  final int warningLevel;
  final int riskScore;
  final bool isFakeVoice;
  final String explanation;
  final DateTime timestamp;
  final Duration duration;

  CallRecord({
    required this.id,
    required this.text,
    required this.warningLevel,
    required this.riskScore,
    required this.isFakeVoice,
    required this.explanation,
    required this.timestamp,
    required this.duration,
  });

  factory CallRecord.fromResult(AnalysisResult result, Duration duration) {
    return CallRecord(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      text: result.text,
      warningLevel: result.warningLevel,
      riskScore: result.riskScore,
      isFakeVoice: result.isFakeVoice,
      explanation: result.explanation,
      timestamp: DateTime.now(),
      duration: duration,
    );
  }

  String get levelLabel => ['안전', '주의', '경고', '위험'][warningLevel];
  String get durationString {
    final m = duration.inMinutes;
    final s = duration.inSeconds % 60;
    return '${m}분 ${s.toString().padLeft(2, '0')}초';
  }
}
