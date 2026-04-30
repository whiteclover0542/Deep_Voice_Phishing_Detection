import 'package:flutter/material.dart';
import '../models/call_record.dart';

class HomeScreen extends StatelessWidget {
  final List<CallRecord> records;
  final bool isProtectionOn;
  final VoidCallback onToggle;
  const HomeScreen({super.key, required this.records, required this.isProtectionOn, required this.onToggle});

  int get _detected => records.where((r) => r.warningLevel >= 1).length;
  int get _blocked => records.where((r) => r.warningLevel >= 2).length;
  int get _safe => records.where((r) => r.warningLevel == 0).length;

  static const _levelColors = [
    Color(0xFF43A047),
    Color(0xFFFF9800),
    Color(0xFFFF9800),
    Color(0xFFE53935),
  ];
  static const _levelIcons = [
    Icons.check_circle,
    Icons.warning_amber,
    Icons.warning_amber,
    Icons.dangerous,
  ];
  static const _levelLabels = ['안전', '주의', '경고', '위험'];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF0F2F5),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: const Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('보이스가드', style: TextStyle(color: Color(0xFF1A1A1A), fontWeight: FontWeight.bold, fontSize: 20)),
            Text('실시간 보이스 피싱 탐지', style: TextStyle(color: Colors.grey, fontSize: 12)),
          ],
        ),
        actions: [
          IconButton(
            icon: Stack(children: [
              const Icon(Icons.notifications_outlined, color: Color(0xFF1A1A1A)),
              if (_detected > 0)
                Positioned(
                  right: 0, top: 0,
                  child: Container(width: 8, height: 8,
                      decoration: const BoxDecoration(color: Colors.red, shape: BoxShape.circle)),
                ),
            ]),
            onPressed: () {},
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _StatusCard(isOn: isProtectionOn, onToggle: onToggle, totalCalls: records.length),
          const SizedBox(height: 16),
          _StatsRow(detected: _detected, blocked: _blocked, safe: _safe),
          const SizedBox(height: 16),
          _WeeklyChart(records: records),
          const SizedBox(height: 16),
          _RecentHistory(records: records.take(5).toList(), levelColors: _levelColors, levelIcons: _levelIcons, levelLabels: _levelLabels),
          const SizedBox(height: 16),
        ],
      ),
    );
  }
}

class _StatusCard extends StatelessWidget {
  final bool isOn;
  final VoidCallback onToggle;
  final int totalCalls;

  const _StatusCard({required this.isOn, required this.onToggle, required this.totalCalls});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF1E5FD8), Color(0xFF2979FF)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(12)),
              child: const Icon(Icons.shield_outlined, color: Colors.white, size: 24),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('보호 상태', style: TextStyle(color: Colors.white70, fontSize: 12)),
                Text(isOn ? '보호 중' : '대기 중', style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
              ]),
            ),
            Switch(
              value: isOn,
              onChanged: (_) => onToggle(),
              activeColor: Colors.white,
              activeTrackColor: Colors.greenAccent,
              inactiveThumbColor: Colors.white,
              inactiveTrackColor: Colors.white38,
            ),
          ]),
          const SizedBox(height: 12),
          Row(children: [
            Container(width: 8, height: 8,
                decoration: BoxDecoration(color: isOn ? Colors.greenAccent : Colors.grey, shape: BoxShape.circle)),
            const SizedBox(width: 6),
            Text(isOn ? 'AI 분석 엔진 활성화됨 · 총 $totalCalls건 분석' : 'AI 분석 엔진 대기 중',
                style: const TextStyle(color: Colors.white70, fontSize: 13)),
          ]),
        ],
      ),
    );
  }
}

class _StatsRow extends StatelessWidget {
  final int detected, blocked, safe;
  const _StatsRow({required this.detected, required this.blocked, required this.safe});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      _StatCard(icon: Icons.search, iconColor: const Color(0xFF1E5FD8), count: detected, label: '오늘 탐지'),
      const SizedBox(width: 10),
      _StatCard(icon: Icons.block, iconColor: Colors.redAccent, count: blocked, label: '차단 완료'),
      const SizedBox(width: 10),
      _StatCard(icon: Icons.check_circle_outline, iconColor: Colors.green, count: safe, label: '안전 통화'),
    ]);
  }
}

class _StatCard extends StatelessWidget {
  final IconData icon;
  final Color iconColor;
  final int count;
  final String label;

  const _StatCard({required this.icon, required this.iconColor, required this.count, required this.label});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14)),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(icon, color: iconColor, size: 20),
          const SizedBox(height: 8),
          Text('$count건', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Color(0xFF1A1A1A))),
          Text(label, style: const TextStyle(color: Colors.grey, fontSize: 11)),
        ]),
      ),
    );
  }
}

class _WeeklyChart extends StatelessWidget {
  final List<CallRecord> records;
  const _WeeklyChart({required this.records});

  @override
  Widget build(BuildContext context) {
    final days = ['월', '화', '수', '목', '금', '토', '일'];
    final counts = List.filled(7, 0);
    final today = DateTime.now().weekday - 1;

    for (final r in records) {
      if (r.warningLevel >= 1) counts[r.timestamp.weekday - 1]++;
    }
    final maxCount = counts.reduce((a, b) => a > b ? a : b).clamp(1, 999);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            Text('주간 탐지 현황', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
            Text('이번 주', style: TextStyle(color: Colors.grey, fontSize: 12)),
          ]),
          const SizedBox(height: 16),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: List.generate(7, (i) {
              final ratio = counts[i] / maxCount;
              final isToday = i == today;
              return Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 3),
                  child: Column(children: [
                    Container(
                      height: 60 * ratio + 4,
                      decoration: BoxDecoration(
                        color: isToday ? const Color(0xFF1E5FD8) : const Color(0xFFE8EDF5),
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(days[i], style: TextStyle(fontSize: 11, color: isToday ? const Color(0xFF1E5FD8) : Colors.grey)),
                  ]),
                ),
              );
            }),
          ),
        ],
      ),
    );
  }
}

class _RecentHistory extends StatelessWidget {
  final List<CallRecord> records;
  final List<Color> levelColors;
  final List<IconData> levelIcons;
  final List<String> levelLabels;

  const _RecentHistory({required this.records, required this.levelColors, required this.levelIcons, required this.levelLabels});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          Text('최근 탐지 이력', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
          Text('전체 보기 >', style: TextStyle(color: Color(0xFF1E5FD8), fontSize: 12)),
        ]),
        const SizedBox(height: 8),
        records.isEmpty
            ? Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14)),
                child: const Center(child: Text('탐지 이력이 없습니다', style: TextStyle(color: Colors.grey))),
              )
            : Container(
                decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14)),
                child: ListView.separated(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  itemCount: records.length,
                  separatorBuilder: (_, __) => const Divider(height: 1, indent: 56),
                  itemBuilder: (_, i) {
                    final r = records[i];
                    final color = levelColors[r.warningLevel];
                    return ListTile(
                      leading: Icon(levelIcons[r.warningLevel], color: color, size: 28),
                      title: Row(children: [
                        Expanded(child: Text(r.text, maxLines: 1, overflow: TextOverflow.ellipsis,
                            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500))),
                        const SizedBox(width: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                          decoration: BoxDecoration(color: color.withOpacity(0.12), borderRadius: BorderRadius.circular(8)),
                          child: Text(levelLabels[r.warningLevel], style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.bold)),
                        ),
                      ]),
                      subtitle: Text('${r.durationString} · ${r.timestamp.month}.${r.timestamp.day}',
                          style: const TextStyle(color: Colors.grey, fontSize: 11)),
                      trailing: const Icon(Icons.chevron_right, color: Colors.grey, size: 18),
                    );
                  },
                ),
              ),
      ],
    );
  }
}
