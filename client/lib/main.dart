import 'package:flutter/material.dart';
import 'models/call_record.dart';
import 'screens/home_screen.dart';
import 'screens/realtime_screen.dart';
import 'screens/history_screen.dart';

void main() {
  runApp(const VoicePhishingApp());
}

class VoicePhishingApp extends StatelessWidget {
  const VoicePhishingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '보이스가드',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: const Color(0xFFF0F2F5),
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1E5FD8)),
        fontFamily: 'sans-serif',
      ),
      home: const MainShell(),
    );
  }
}

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _currentIndex = 0;
  final List<CallRecord> _records = [];
  bool _isProtectionOn = false;

  void _onCallEnded(CallRecord record) {
    setState(() => _records.insert(0, record));
  }

  void _toggleProtection() {
    setState(() => _isProtectionOn = !_isProtectionOn);
  }

  @override
  Widget build(BuildContext context) {
    final screens = [
      HomeScreen(records: _records, isProtectionOn: _isProtectionOn, onToggle: _toggleProtection),
      RealtimeScreen(onCallEnded: _onCallEnded, isProtectionOn: _isProtectionOn),
      HistoryScreen(records: _records),
    ];

    return Scaffold(
      body: SafeArea(child: screens[_currentIndex]),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (i) => setState(() => _currentIndex = i),
        backgroundColor: Colors.white,
        selectedItemColor: const Color(0xFF1E5FD8),
        unselectedItemColor: Colors.grey,
        selectedLabelStyle: const TextStyle(fontWeight: FontWeight.bold, fontSize: 11),
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home_outlined), activeIcon: Icon(Icons.home), label: '홈'),
          BottomNavigationBarItem(icon: Icon(Icons.phone_outlined), activeIcon: Icon(Icons.phone), label: '실시간'),
          BottomNavigationBarItem(icon: Icon(Icons.history_outlined), activeIcon: Icon(Icons.history), label: '이력'),
        ],
      ),
    );
  }
}
