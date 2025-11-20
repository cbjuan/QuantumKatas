# Dataset Files Reference

**Last Updated:** 2025-11-20 (v3)
**Dataset Version:** quantum_katas_dataset.jsonl v3 (71.4% passing)

---

## Essential Files (11 total)

### Core Dataset (2 files)

| File | Size | Purpose |
|------|------|---------|
| **quantum_katas_dataset.jsonl** | 945KB | Production dataset (v3, 250/350 passing) |
| **quantum_katas_dataset_original.jsonl** | 956KB | Original backup (0/350 passing) |

### Tools (3 files)

| File | Size | Purpose |
|------|------|---------|
| **validate_dataset.py** | 4.6KB | Validates all entries, runs tests |
| **analyze_failures.py** | 5.4KB | Categorizes failures by type |
| **fix_dataset_comprehensive.py** | 9.0KB | Reference fixer (Pass 1 & 2) |
| **fix_targeted_issues.py** | 11KB | Reference fixer (v3, Pass 3) |

### Documentation (5 files)

| File | Size | Purpose |
|------|------|---------|
| **QISKIT_DATASET.md** | 5.6KB | Quick reference guide - **START HERE** |
| **FINAL_STATUS.md** | 9.0KB | Current status and metrics |
| **IMPROVEMENTS_COMPLETE.md** | 9.1KB | History of v1 & v2 improvements |
| **IMPROVEMENTS_V3.md** | 5.4KB | Latest v3 improvements |
| **failure_analysis_report.txt** | 11KB | Detailed failure breakdown |

### Analysis Report (1 file)

| File | Size | Purpose |
|------|------|---------|
| **failure_analysis_report.txt** | 11KB | Categorized list of 100 remaining failures |

---

## File Usage Guide

### For Users

**Getting Started:**
1. Read [QISKIT_DATASET.md](QISKIT_DATASET.md) - Quick start guide
2. Use `quantum_katas_dataset.jsonl` - Current production data

**Understanding Quality:**
1. Check [FINAL_STATUS.md](FINAL_STATUS.md) - Current metrics
2. Review `failure_analysis_report.txt` - Known issues

### For Developers

**Validating Dataset:**
```bash
python3 validate_dataset.py --file quantum_katas_dataset.jsonl --continue
```

**Analyzing Failures:**
```bash
python3 analyze_failures.py
cat failure_analysis_report.txt
```

**Understanding Fixes:**
1. Read [IMPROVEMENTS_COMPLETE.md](IMPROVEMENTS_COMPLETE.md) - v1 & v2 history
2. Read [IMPROVEMENTS_V3.md](IMPROVEMENTS_V3.md) - Latest improvements
3. Review `fix_dataset_comprehensive.py` - Pass 1 & 2 methodology
4. Review `fix_targeted_issues.py` - Pass 3 methodology

### For Future Improvements

**Reference Scripts:**
- `fix_dataset_comprehensive.py` - Shows how to fix API compatibility issues
- `fix_targeted_issues.py` - Shows how to fix simulator-related issues

**Known Issues:**
- See `failure_analysis_report.txt` for categorized list of 100 remaining failures

---

## Version History

### v3 (Current) - 2025-11-20
- **Pass Rate:** 250/350 (71.4%)
- **Changes:** Fixed 6 simulator-related issues
- **Tool:** fix_targeted_issues.py

### v2 - 2025-11-20
- **Pass Rate:** 244/350 (69.7%)
- **Changes:** Fixed type mismatches, deprecated APIs
- **Tool:** fix_dataset_comprehensive.py (Pass 2)

### v1 - 2025-11-20
- **Pass Rate:** 210/350 (60%)
- **Changes:** Fixed critical API compatibility
- **Tool:** fix_dataset_comprehensive.py (Pass 1)

### v0 (Original) - Before fixes
- **Pass Rate:** 0/350 (0%)
- **Status:** Completely broken, incompatible with Qiskit 2.x

---

## Files Removed

### Removed in v3 Cleanup
- `quantum_katas_dataset_backup_v2.jsonl` - Duplicate of v2
- `quantum_katas_dataset_targeted.jsonl` - Duplicate of current
- `CLEANUP_SUMMARY.md` - Outdated

### Removed in v2 Cleanup
- Various intermediate fix scripts (fix_dataset.py, fix_dataset_v2.py, etc.)
- Intermediate dataset versions (quantum_katas_dataset_fixed.jsonl, etc.)
- Intermediate logs and test utilities
- Redundant documentation (DATASET_REVIEW.md, FIX_SUMMARY.md, FINAL_REPORT.md)

---

## Disk Usage

**Total Essential Files:** ~1.0 MB
- Datasets: 1.9 MB (2 files)
- Tools: 30 KB (4 files)
- Documentation: 40 KB (5 files)

---

## Recommendations

### For Production Use
✅ Use: `quantum_katas_dataset.jsonl` (v3)
✅ Filter: Use 6 perfect categories or 12 production-ready categories
✅ Validate: Run `validate_dataset.py` to verify

### For Development
✅ Keep: `quantum_katas_dataset_original.jsonl` for rollback
✅ Reference: Fix scripts show methodology for future improvements
✅ Monitor: `failure_analysis_report.txt` tracks known issues

### For Understanding
✅ Start: QISKIT_DATASET.md
✅ Deep Dive: IMPROVEMENTS_COMPLETE.md + IMPROVEMENTS_V3.md
✅ Current State: FINAL_STATUS.md

---

**Clean, organized, and production-ready!** ✅
