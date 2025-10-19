"""
Create comprehensive department mapping for all 577 diseases

This script creates a complete disease-to-department mapping using:
1. Existing manual mappings from old system
2. Pattern-based automatic mapping using medical keywords
3. Default fallback for unmapped diseases

Author: Medical Chatbot Team
Date: 2025-10-18
"""

import json
import re
from pathlib import Path

# Define all hospital departments
# Order matters - more specific patterns first!
DEPARTMENTS = {
    # Very specific conditions first
    "Ung bướu": ["ung thư", "ung thu", "carcinoma", "lymphoma", "leukemia"],
    "Cấp cứu": ["xuất huyết", "tràn khí", "co giật", "ngừng tim", "ngất xỉu", "đột quỵ"],
    "Chấn thương chỉnh hình": ["gãy xương", "rách", "bong gân", "chấn thương", "trật khớp"],

    # Specific organ systems
    "Nội hô hấp": ["phổi", "hen suyễn", "hen phế quản", "phế quản", "lao phổi", "ho", "khò khè", "covid", "viêm phổi"],
    "Tim mạch": ["tim", "nhịp tim", "huyết áp", "động mạch", "tĩnh mạch", "block tim", "mạch máu"],
    "Tiêu hoá – Gan mật": ["gan", "mật", "túi mật", "viêm gan", "xơ gan", "gan to"],
    "Tiêu hoá": ["dạ dày", "ruột", "loét dạ dày", "đại tràng", "táo bón", "tiêu chảy", "phân", "hậu môn"],
    "Thận – Tiết niệu": ["thận", "tiết niệu", "bàng quang", "sỏi thận", "viêm thận"],
    "Tai Mũi Họng": ["tai", "mũi", "họng", "thanh quản", "amiđan", "xoang", "amidan", "thính lực", "điếc"],
    "Mắt": ["mắt", "võng mạc", "thị lực", "nhãn cầu", "giác mạc", "mù", "đục thủy tinh"],
    "Thần kinh": ["não", "alzheimer", "parkinson", "đau nửa đầu", "chóng mặt", "động kinh", "liệt", "bại não", "sa sút trí tuệ"],
    "Cơ xương khớp": ["xương", "khớp", "viêm khớp", "gout", "loãng xương", "thoái hóa khớp", "cột sống"],
    "Da liễu": ["da", "mụn", "nấm da", "viêm da", "chàm", "vẩy nến", "mề đay", "eczema", "psoriasis", "ghẻ"],
    "Sản phụ khoa": ["âm đạo", "tử cung", "buồng trứng", "vú", "kinh nguyệt", "thai kỳ", "mang thai"],
    "Nội tiết": ["tiểu đường", "đái tháo đường", "tuyến giáp", "nội tiết", "hormone", "cường giáp", "suy giáp"],
    "Truyền nhiễm": ["sốt xuất huyết", "sốt rét", "hiv", "aids", "thủy đậu", "thương hàn", "viêm gan virus"],
    "Huyết học": ["thiếu máu", "bạch cầu", "hồng cầu", "đông máu", "máu khó đông"],
    "Nam khoa": ["tuyến tiền liệt", "tinh hoàn", "dương vật", "bìu", "tinh trùng"],
    "Tâm thần": ["trầm cảm", "lo âu", "tâm thần", "tự kỷ", "tăng động giảm chú ý", "rối loạn lưỡng cực"],
    "Ngoại tổng quát": ["ruột thừa", "thoát vị", "polyp"],
    "Răng Hàm Mặt": ["răng", "nướu", "implant", "sâu răng", "viêm nướu"],
    "Khám tổng quát": []  # Default fallback
}

def load_existing_mapping():
    """Load existing department mapping from old system"""
    old_map_path = Path(__file__).parent / "department_map_old.json"

    if old_map_path.exists():
        with open(old_map_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def match_department(disease_name, existing_map):
    """
    Match disease to department using:
    1. Exact match in existing mapping
    2. Case-insensitive match in existing mapping
    3. Pattern-based keyword matching
    4. Default fallback
    """
    # 1. Exact match
    if disease_name in existing_map:
        return existing_map[disease_name]

    # 2. Case-insensitive match
    disease_lower = disease_name.lower().strip()
    for k, v in existing_map.items():
        if k.lower().strip() == disease_lower:
            return v

    # 3. Pattern-based matching
    for dept, keywords in DEPARTMENTS.items():
        if dept == "Khám tổng quát":  # Skip default
            continue

        for keyword in keywords:
            if keyword in disease_lower:
                return dept

    # 4. Default fallback
    return "Khám tổng quát"

def create_comprehensive_mapping(diseases_file):
    """Create complete mapping for all 577 diseases"""
    # Load existing mapping
    existing_map = load_existing_mapping()
    print(f"Loaded {len(existing_map)} existing mappings")

    # Load all diseases
    with open(diseases_file, 'r', encoding='utf-8') as f:
        diseases = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(diseases)} diseases...")

    # Create new mapping
    new_mapping = {}
    dept_counts = {}

    for disease in diseases:
        dept = match_department(disease, existing_map)
        new_mapping[disease] = dept
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

    return new_mapping, dept_counts

def main():
    """Main function"""
    script_dir = Path(__file__).parent
    diseases_file = script_dir / "all_577_diseases.txt"
    output_file = script_dir / "department_map_577.json"

    print("="*80)
    print("CREATING DEPARTMENT MAPPING FOR 577 DISEASES")
    print("="*80)

    # Create mapping
    mapping, counts = create_comprehensive_mapping(diseases_file)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Mapping saved to: {output_file}")
    print(f"\n📊 Department Distribution:")
    print("-"*80)

    # Sort by count
    for dept, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(mapping) * 100
        bar = "█" * int(pct / 2)
        print(f"  {dept:30s} {count:3d} ({pct:5.1f}%) {bar}")

    print(f"\n{'='*80}")
    print(f"Total diseases mapped: {len(mapping)}")

    # Show sample mappings
    print(f"\n📋 Sample mappings (first 20):")
    print("-"*80)
    for i, (disease, dept) in enumerate(list(mapping.items())[:20], 1):
        print(f"  {i:2d}. {disease:50s} → {dept}")

    print(f"\n💡 Review the mapping and manually adjust any incorrect assignments")
    print(f"   File: {output_file}")

if __name__ == "__main__":
    main()
