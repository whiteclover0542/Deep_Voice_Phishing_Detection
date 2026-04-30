import 'package:flutter/material.dart';
import '../models/call_record.dart';

class HistoryScreen extends StatefulWidget {
  final List<CallRecord> records;

  const HistoryScreen({super.key, required this.records});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  String _searchQuery = '';
  int _selectedFilter = 0; // 0: 전체, 1: 위험, 2: 주의, 3: 안전

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

  List<CallRecord> get _filtered {
    return widget.records.where((r) {
      final matchSearch = _searchQuery.isEmpty || r.text.contains(_searchQuery);
      final matchFilter = _selectedFilter == 0 ||
          (_selectedFilter == 1 && r.warningLevel == 3) ||
          (_selectedFilter == 2 && (r.warningLevel == 1 || r.warningLevel == 2)) ||
          (_selectedFilter == 3 && r.warningLevel == 0);
      return matchSearch && matchFilter;
    }).toList();
  }

  int _countByLevel(int level) => widget.records.where((r) {
        if (level == 1) return r.warningLevel == 3;
        if (level == 2) return r.warningLevel == 1 || r.warningLevel == 2;
        return r.warningLevel == 0;
      }).length;

  @override
  Widget build(BuildContext context) {
    final filtered = _filtered;
    final dangerCount = _countByLevel(1);
    final cautionCount = _countByLevel(2);
    final safeCount = _countByLevel(3);

    return Column(
      children: [
        // 헤더
        Container(
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
              Text('통화 이력', style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
              SizedBox(height: 4),
              Text('AI 분석 결과가 저장됩니다', style: TextStyle(color: Colors.white70, fontSize: 13)),
            ],
          ),
        ),

        Expanded(
          child: widget.records.isEmpty
              ? _buildEmpty()
              : Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                      child: Column(
                        children: [
                          // 검색
                          TextField(
                            onChanged: (v) => setState(() => _searchQuery = v),
                            decoration: InputDecoration(
                              hintText: '번호, 키워드로 검색...',
                              hintStyle: const TextStyle(color: Colors.grey, fontSize: 13),
                              prefixIcon: const Icon(Icons.search, color: Colors.grey, size: 20),
                              filled: true,
                              fillColor: Colors.white,
                              contentPadding: const EdgeInsets.symmetric(vertical: 0),
                              border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                            ),
                          ),
                          const SizedBox(height: 12),

                          // 통계
                          Row(children: [
                            _StatBadge(count: dangerCount, label: '위험', color: const Color(0xFFE53935)),
                            const SizedBox(width: 10),
                            _StatBadge(count: cautionCount, label: '주의', color: const Color(0xFFFF9800)),
                            const SizedBox(width: 10),
                            _StatBadge(count: safeCount, label: '안전', color: const Color(0xFF43A047)),
                          ]),
                          const SizedBox(height: 12),

                          // 필터 탭
                          SingleChildScrollView(
                            scrollDirection: Axis.horizontal,
                            child: Row(
                              children: [
                                _FilterChip(label: '전체 ${widget.records.length}', selected: _selectedFilter == 0, onTap: () => setState(() => _selectedFilter = 0)),
                                const SizedBox(width: 8),
                                _FilterChip(label: '위험 $dangerCount', selected: _selectedFilter == 1, onTap: () => setState(() => _selectedFilter = 1)),
                                const SizedBox(width: 8),
                                _FilterChip(label: '주의 $cautionCount', selected: _selectedFilter == 2, onTap: () => setState(() => _selectedFilter = 2)),
                                const SizedBox(width: 8),
                                _FilterChip(label: '안전 $safeCount', selected: _selectedFilter == 3, onTap: () => setState(() => _selectedFilter = 3)),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),

                    // 목록
                    Expanded(
                      child: filtered.isEmpty
                          ? const Center(child: Text('검색 결과가 없습니다', style: TextStyle(color: Colors.grey)))
                          : ListView.separated(
                              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                              itemCount: filtered.length,
                              separatorBuilder: (_, __) => const SizedBox(height: 8),
                              itemBuilder: (_, i) => _RecordTile(
                                record: filtered[i],
                                levelColor: _levelColors[filtered[i].warningLevel],
                                levelIcon: _levelIcons[filtered[i].warningLevel],
                              ),
                            ),
                    ),
                  ],
                ),
        ),
      ],
    );
  }

  Widget _buildEmpty() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.history, size: 60, color: Colors.grey),
          const SizedBox(height: 12),
          const Text('통화 이력이 없습니다', style: TextStyle(color: Colors.grey, fontSize: 15)),
          const SizedBox(height: 4),
          const Text('실시간 탭에서 통화 감지를 시작해보세요', style: TextStyle(color: Colors.grey, fontSize: 12)),
        ],
      ),
    );
  }
}

class _StatBadge extends StatelessWidget {
  final int count;
  final String label;
  final Color color;

  const _StatBadge({required this.count, required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(color: color.withOpacity(0.08), borderRadius: BorderRadius.circular(12)),
        child: Column(children: [
          Text('$count', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: color)),
          Text(label, style: TextStyle(fontSize: 12, color: color)),
        ]),
      ),
    );
  }
}

class _FilterChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _FilterChip({required this.label, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? const Color(0xFF1A1A1A) : Colors.white,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(label, style: TextStyle(color: selected ? Colors.white : Colors.grey, fontSize: 13, fontWeight: selected ? FontWeight.bold : FontWeight.normal)),
      ),
    );
  }
}

class _RecordTile extends StatefulWidget {
  final CallRecord record;
  final Color levelColor;
  final IconData levelIcon;

  const _RecordTile({required this.record, required this.levelColor, required this.levelIcon});

  @override
  State<_RecordTile> createState() => _RecordTileState();
}

class _RecordTileState extends State<_RecordTile> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final r = widget.record;
    return Container(
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14)),
      child: Column(
        children: [
          ListTile(
            leading: Icon(widget.levelIcon, color: widget.levelColor, size: 28),
            title: Row(children: [
              Expanded(child: Text(r.text, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600), maxLines: 1, overflow: TextOverflow.ellipsis)),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(color: widget.levelColor.withOpacity(0.12), borderRadius: BorderRadius.circular(8)),
                child: Text(r.levelLabel, style: TextStyle(color: widget.levelColor, fontSize: 11, fontWeight: FontWeight.bold)),
              ),
            ]),
            subtitle: Text('${r.durationString} · ${r.timestamp.month}.${r.timestamp.day} ${r.timestamp.hour.toString().padLeft(2,'0')}:${r.timestamp.minute.toString().padLeft(2,'0')}',
                style: const TextStyle(color: Colors.grey, fontSize: 11)),
            trailing: GestureDetector(
              onTap: () => setState(() => _expanded = !_expanded),
              child: Icon(_expanded ? Icons.expand_less : Icons.expand_more, color: Colors.grey),
            ),
          ),
          if (_expanded)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(color: const Color(0xFFF5F7FA), borderRadius: BorderRadius.circular(10)),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  const Text('AI 분석', style: TextStyle(color: Colors.grey, fontSize: 11)),
                  const SizedBox(height: 4),
                  Text(r.explanation, style: const TextStyle(fontSize: 13, color: Color(0xFF1A1A1A))),
                  const SizedBox(height: 8),
                  Text('위험 점수: ${r.riskScore}/100 · ${r.isFakeVoice ? "합성 음성 의심" : "정상 음성"}',
                      style: const TextStyle(fontSize: 11, color: Colors.grey)),
                ]),
              ),
            ),
        ],
      ),
    );
  }
}
