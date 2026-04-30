import 'dart:async';
import 'package:record/record.dart';

class AudioService {
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<List<int>>? _subscription;

  Future<bool> hasPermission() async {
    return await _recorder.hasPermission();
  }

  Future<void> start({required Function(List<int>) onAudioChunk}) async {
    final stream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
      ),
    );

    _subscription = stream.listen(
      (chunk) => onAudioChunk(chunk),
      onError: (_) => stop(),
    );
  }

  Future<void> stop() async {
    await _subscription?.cancel();
    _subscription = null;
    await _recorder.stop();
  }

  void dispose() {
    _recorder.dispose();
  }
}
