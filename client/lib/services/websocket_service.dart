import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../models/analysis_result.dart';

class WebSocketService {
  static const String _serverUrl = 'ws://localhost:8000/ws/audio';

  WebSocketChannel? _channel;
  bool _isConnected = false;

  bool get isConnected => _isConnected;

  void connect({required Function(AnalysisResult) onResult, required Function() onDisconnected}) {
    _channel = WebSocketChannel.connect(Uri.parse(_serverUrl));
    _isConnected = true;

    _channel!.stream.listen(
      (message) {
        final json = jsonDecode(message as String) as Map<String, dynamic>;
        onResult(AnalysisResult.fromJson(json));
      },
      onDone: () {
        _isConnected = false;
        onDisconnected();
      },
      onError: (_) {
        _isConnected = false;
        onDisconnected();
      },
    );
  }

  void sendAudio(List<int> audioBytes) {
    if (_isConnected && _channel != null) {
      _channel!.sink.add(audioBytes);
    }
  }

  void disconnect() {
    _channel?.sink.close();
    _isConnected = false;
  }
}
