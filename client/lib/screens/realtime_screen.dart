import 'package:flutter/material.dart';
import '../models/analysis_result.dart';
import '../models/call_record.dart';
import '../services/audio_service.dart';
import '../services/websocket_service.dart';

enum _CallState { idle, incoming, active, ended }

class RealtimeScreen extends StatefulWidget {
  final Function(CallRecord) onCallEnded;
  final bool isProtectionOn;

  const RealtimeScreen({super.key, required this.onCallEnded, required this.isProtectionOn});

  @override
  State<RealtimeScreen> createState() => _RealtimeScreenState();
}

class _RealtimeScreenState extends State<RealtimeScreen> {
  final _wsService = WebSocketService();
  final _audioService = AudioService();

  _CallState _callState = _CallState.idle;
  AnalysisResult? _result;
  DateTime? _startTime;

  static const _levelColors = [
    Color(0xFF43A047),
    Color(0xFFFF9800),
    Color(0xFFFF9800),
    Color(0xFFE53935),
  ];
  static const _levelLabels = ['안전', '주의', '경고', '위험'];

  // 시뮬레이션: 전화 수신
  void _simulateIncomingCall() {
    setState(() => _callState = _CallState.incoming);
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted && _callState == _CallState.incoming) _startDetection();
    });
  }

  Future<void> _startDetection() async {
    final hasPermission = await _audioService.hasPermission();
    if (!hasPermission) return;

    _wsService.connect(
      onResult: (result) => setState(() => _result = result),
      onDisconnected: () => setState(() => _callState = _CallState.idle),
    );

    await _audioService.start(
      onAudioChunk: (chunk) => _wsService.sendAudio(chunk),
    );

    setState(() {
      _callState = _CallState.active;
      _startTime = DateTime.now();
    });
  }

  Future<void> _endCall() async {
    await _audioService.stop();
    _wsService.disconnect();

    if (_result != null && _startTime != null) {
      widget.onCallEnded(CallRecord.fromResult(_result!, DateTime.now().difference(_startTime!)));
    }

    setState(() {
      _callState = _CallState.ended;
      _result = null;
    });

    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => _callState = _CallState.idle);
    });
  }

  @override
  void dispose() {
    _audioService.dispose();
    _wsService.disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final level = _callState == _CallState.active ? (_result?.warningLevel ?? 0) : 0;

    return Column(
      children: [
        // 헤더 (위험도에 따라 변화)
        _buildHeader(level),

        // 1단계: 주의 배너
        if (_callState == _CallState.active && level == 1)
          _WarningBanner(
            color: const Color(0xFFFFF8E1),
            borderColor: const Color(0xFFFFEB3B),
            icon: Icons.warning_amber,
            iconColor: const Color(0xFFF9A825),
            message: '주의: 보이스피싱 패턴이 감지되었습니다',
          ),

        // 2단계: 경고 배너
        if (_callState == _CallState.active && level == 2)
          _WarningBanner(
            color: const Color(0xFFFFF3E0),
            borderColor: const Color(0xFFFF9800),
            icon: Icons.error_outline,
            iconColor: const Color(0xFFE65100),
            message: '경고: 보이스피싱 가능성이 높습니다!',
          ),

        Expanded(
          child: Container(
            decoration: BoxDecoration(
              border: _callState == _CallState.active && level == 2
                  ? Border.all(color: const Color(0xFFFF9800), width: 2)
                  : null,
            ),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: _buildBody(),
            ),
          ),
        ),

        _buildBottomArea(),
      ],
    );
  }

  Widget _buildHeader(int level) {
    // 3단계 위험: 상단 빨간색으로 채움
    if (_callState == _CallState.active && level == 3) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
        color: const Color(0xFFE53935),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: const [
              Icon(Icons.dangerous, color: Colors.white, size: 28),
              SizedBox(width: 8),
              Text('위험 감지!', style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
            ]),
            const SizedBox(height: 4),
            const Text('보이스피싱이 강하게 의심됩니다. 즉시 전화를 끊으세요!',
                style: TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w500)),
          ],
        ),
      );
    }

    // 기본 헤더 (파란색)
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF1E3A6E), Color(0xFF1E5FD8)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('실시간 통화 감지', style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
          SizedBox(height: 4),
          Text('AI가 통화 내용을 실시간으로 분석합니다', style: TextStyle(color: Colors.white70, fontSize: 13)),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (!widget.isProtectionOn) return _buildProtectionOff();

    switch (_callState) {
      case _CallState.idle:
        return _buildIdle();
      case _CallState.incoming:
        return _buildIncoming();
      case _CallState.active:
        return _buildActive();
      case _CallState.ended:
        return _buildEnded();
    }
  }

  Widget _buildProtectionOff() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: const Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.shield_outlined, size: 48, color: Colors.grey),
          SizedBox(height: 16),
          Text('보호가 비활성화되어 있습니다', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF1A1A1A))),
          SizedBox(height: 8),
          Text('홈 화면에서 보호 기능을 켜주세요', style: TextStyle(color: Colors.grey, fontSize: 13)),
        ],
      ),
    );
  }

  Widget _buildIdle() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: const BoxDecoration(color: Color(0xFFF0F2F5), shape: BoxShape.circle),
            child: const Icon(Icons.phone_outlined, size: 40, color: Colors.grey),
          ),
          const SizedBox(height: 16),
          const Text('대기 중', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Color(0xFF1A1A1A))),
          const SizedBox(height: 8),
          const Text('전화가 오면 자동으로 분석이 시작됩니다', style: TextStyle(color: Colors.grey, fontSize: 13)),
        ],
      ),
    );
  }

  Widget _buildIncoming() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: Colors.greenAccent.withOpacity(0.15), shape: BoxShape.circle),
            child: const Icon(Icons.phone_in_talk, size: 40, color: Colors.green),
          ),
          const SizedBox(height: 16),
          const Text('전화 수신 중...', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Color(0xFF1A1A1A))),
          const SizedBox(height: 8),
          const Text('AI 분석 엔진이 활성화되고 있습니다', style: TextStyle(color: Colors.grey, fontSize: 13)),
          const SizedBox(height: 16),
          const CircularProgressIndicator(color: Color(0xFF1E5FD8)),
        ],
      ),
    );
  }

  Widget _buildEnded() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: const Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.phone_disabled, size: 48, color: Colors.grey),
          SizedBox(height: 16),
          Text('통화 종료', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Color(0xFF1A1A1A))),
          SizedBox(height: 8),
          Text('분석 결과가 이력에 저장되었습니다', style: TextStyle(color: Colors.grey, fontSize: 13)),
        ],
      ),
    );
  }

  Widget _buildActive() {
    final level = _result?.warningLevel ?? 0;
    final score = _result?.riskScore ?? 0;
    final levelColor = _levelColors[level];

    return SingleChildScrollView(
      child: Column(
        children: [
          // 위험 점수 + 상태
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
            child: Row(children: [
              Container(
                width: 80, height: 80,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: levelColor, width: 4),
                  color: levelColor.withOpacity(0.08),
                ),
                child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                  Text('$score', style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold, color: levelColor)),
                  Text('/100', style: TextStyle(fontSize: 10, color: levelColor.withOpacity(0.7))),
                ]),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  const Text('위험 점수', style: TextStyle(color: Colors.grey, fontSize: 12)),
                  const SizedBox(height: 4),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(color: levelColor.withOpacity(0.12), borderRadius: BorderRadius.circular(8)),
                    child: Text(_levelLabels[level], style: TextStyle(color: levelColor, fontWeight: FontWeight.bold)),
                  ),
                  const SizedBox(height: 8),
                  Row(children: [
                    Icon(Icons.circle, size: 8, color: (_result?.isFakeVoice ?? false) ? Colors.orange : Colors.green),
                    const SizedBox(width: 4),
                    Text((_result?.isFakeVoice ?? false) ? '합성 음성 의심' : '정상 음성',
                        style: const TextStyle(fontSize: 12, color: Colors.grey)),
                  ]),
                ]),
              ),
            ]),
          ),
          const SizedBox(height: 12),
          _ResultCard(icon: Icons.text_fields, title: '통화 내용', content: _result?.text ?? '분석 중...'),
          const SizedBox(height: 12),
          _ResultCard(icon: Icons.auto_awesome, title: 'AI 분석', content: _result?.explanation ?? '분석 중...', contentColor: const Color(0xFF1E5FD8)),
        ],
      ),
    );
  }

  Widget _buildBottomArea() {
    if (!widget.isProtectionOn) {
      return const Padding(
        padding: EdgeInsets.only(bottom: 16),
        child: Text('홈에서 보호 기능을 켜주세요', style: TextStyle(color: Colors.grey, fontSize: 12)),
      );
    }

    switch (_callState) {
      case _CallState.idle:
        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            SizedBox(
              width: double.infinity, height: 52,
              child: ElevatedButton.icon(
                onPressed: _simulateIncomingCall,
                icon: const Icon(Icons.phone_in_talk),
                label: const Text('시뮬레이션: 전화 수신', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF1E5FD8),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                  elevation: 0,
                ),
              ),
            ),
            const SizedBox(height: 8),
            const Text('※ 데모용입니다. 실제 앱에서는 전화 수신 시 자동 실행됩니다.',
                style: TextStyle(color: Colors.grey, fontSize: 11)),
          ]),
        );
      case _CallState.active:
        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: SizedBox(
            width: double.infinity, height: 52,
            child: ElevatedButton.icon(
              onPressed: _endCall,
              icon: const Icon(Icons.call_end),
              label: const Text('통화 종료', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.redAccent,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                elevation: 0,
              ),
            ),
          ),
        );
      default:
        return const SizedBox(height: 16);
    }
  }
}

class _WarningBanner extends StatelessWidget {
  final Color color;
  final Color borderColor;
  final IconData icon;
  final Color iconColor;
  final String message;

  const _WarningBanner({
    required this.color,
    required this.borderColor,
    required this.icon,
    required this.iconColor,
    required this.message,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: color,
        border: Border(bottom: BorderSide(color: borderColor, width: 1.5)),
      ),
      child: Row(children: [
        Icon(icon, color: iconColor, size: 18),
        const SizedBox(width: 8),
        Text(message, style: TextStyle(color: iconColor, fontSize: 13, fontWeight: FontWeight.w600)),
      ]),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String content;
  final Color? contentColor;

  const _ResultCard({required this.icon, required this.title, required this.content, this.contentColor});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(icon, size: 16, color: Colors.grey),
          const SizedBox(width: 6),
          Text(title, style: const TextStyle(color: Colors.grey, fontSize: 12)),
        ]),
        const SizedBox(height: 8),
        Text(content, style: TextStyle(fontSize: 14, color: contentColor ?? const Color(0xFF1A1A1A))),
      ]),
    );
  }
}
