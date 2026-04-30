class AnalysisResult {
  final String text;
  final int riskScore;
  final int warningLevel;
  final bool isFakeVoice;
  final double deepfakeConfidence;
  final String explanation;

  AnalysisResult({
    required this.text,
    required this.riskScore,
    required this.warningLevel,
    required this.isFakeVoice,
    required this.deepfakeConfidence,
    required this.explanation,
  });

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      text: json['text'] ?? '',
      riskScore: json['risk_score'] ?? 0,
      warningLevel: json['warning_level'] ?? 0,
      isFakeVoice: json['is_fake_voice'] ?? false,
      deepfakeConfidence: (json['deepfake_confidence'] ?? 0).toDouble(),
      explanation: json['explanation'] ?? '',
    );
  }
}
