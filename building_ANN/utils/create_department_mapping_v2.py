"""
Enhanced department mapping with specific disease mappings and better pattern matching

This version includes:
1. Specific disease-to-department mappings
2. Enhanced pattern matching
3. Better fallback logic

Author: Medical Chatbot Team
Date: 2025-10-18
"""

import json
import re
from pathlib import Path

# Specific disease mappings (high priority)
SPECIFIC_MAPPINGS = {
    # Neurological
    "bệnh Alzheimer": "Thần kinh",
    "bệnh Parkinson": "Thần kinh",
    "bệnh Meniere": "Tai Mũi Họng",
    "bại não": "Thần kinh",
    "bệnh xơ cứng teo cơ một bên (ALS)": "Thần kinh",
    "bệnh đa xơ cứng": "Thần kinh",
    "chứng mất trí nhớ": "Thần kinh",
    "chứng mất trí nhớ thể Lewy": "Thần kinh",
    "chứng ngủ rũ": "Thần kinh",
    "chèn ép thần kinh gần vai": "Thần kinh",

    # Endocrine
    "bệnh Graves": "Nội tiết",
    "bướu cổ": "Nội tiết",
    "bướu cổ đa nhân độc hại": "Nội tiết",

    # GI
    "bệnh Crohn": "Tiêu hoá",
    "bệnh celiac": "Tiêu hoá",
    "bệnh trĩ": "Tiêu hoá",
    "bệnh túi thừa": "Tiêu hoá",
    "chứng khó tiêu": "Tiêu hoá",
    "chứng khó nuốt": "Tai Mũi Họng",
    "chứng nôn nghén": "Tiêu hoá",

    # Infectious/STD
    "bệnh Lyme": "Truyền nhiễm",
    "bệnh chlamydia": "Sản phụ khoa",
    "bệnh lậu": "Sản phụ khoa",
    "bệnh herpes sinh dục": "Sản phụ khoa",
    "bệnh zona (herpes zoster)": "Da liễu",
    "bệnh ký sinh trùng": "Truyền nhiễm",
    "bệnh sán lợn": "Truyền nhiễm",

    # Dermatology
    "bệnh rosacea": "Da liễu",
    "bệnh nấm chân của vận động viên": "Da liễu",
    "bệnh nấm móng": "Da liễu",
    "bệnh vẩy phấn hồng": "Da liễu",
    "chấy rận": "Da liễu",

    # Rheumatology/Autoimmune
    "bệnh lupus ban đỏ hệ thống (SLE)": "Cơ xương khớp",
    "bệnh sarcoidosis": "Nội hô hấp",

    # Cardiology
    "bệnh van hai lá": "Tim mạch",
    "co thắt nhĩ sớm (pacs)": "Tim mạch",
    "co thắt thất sớm (PVCS)": "Tim mạch",

    # Urology/Gynecology
    "bệnh peyronie": "Nam khoa",
    "bệnh viêm vùng chậu": "Sản phụ khoa",

    # Other
    "bệnh nhược cơ": "Thần kinh",
    "bệnh vẹo cổ": "Cơ xương khớp",
    "bệnh de quervain": "Cơ xương khớp",
    "bệnh thần kinh do thuốc": "Thần kinh",
    "bệnh rối loạn tiết mồ hôi": "Da liễu",
    "bệnh đa u tủy": "Huyết học",
    "cai rượu": "Tâm thần",
    "cai thuốc": "Tâm thần",
    "co thắt cơ": "Cơ xương khớp",
    "bệnh cườm nước": "Nội tiết",
    "bệnh cườm nước cấp tính": "Nội tiết",
    "bệnh cườm nước mãn tính": "Nội tiết",
}

# Pattern-based department mapping (ordered by specificity)
DEPARTMENT_PATTERNS = {
    # Very specific first
    "Ung bướu": [
        r"ung thư",
        r"carcinoma",
        r"lymphoma",
        r"leukemia",
        r"u ác tính",
    ],

    "Răng Hàm Mặt": [
        r"\brăng\b",
        r"nướu",
        r"sâu răng",
        r"áp xe răng",
        r"viêm lợi",
        r"viêm nha chu",
    ],

    "Cấp cứu": [
        r"xuất huyết não",
        r"tràn khí màng phổi",
        r"ngừng tim",
        r"đột quỵ",
        r"co giật",
    ],

    "Chấn thương chỉnh hình": [
        r"gãy",
        r"bong gân",
        r"trật khớp",
        r"chấn thương",
        r"vết thương",
    ],

    # Organ-specific
    "Nội hô hấp": [
        r"phổi",
        r"hen\s",
        r"phế quản",
        r"lao phổi",
        r"viêm phổi",
        r"khò khè",
        r"khó thở",
    ],

    "Tim mạch": [
        r"\btim\b",
        r"nhịp tim",
        r"huyết áp",
        r"mạch",
        r"block tim",
        r"suy tim",
        r"van tim",
    ],

    "Tiêu hoá – Gan mật": [
        r"\bgan\b",
        r"mật",
        r"túi mật",
        r"viêm gan",
        r"xơ gan",
    ],

    "Tiêu hoá": [
        r"dạ dày",
        r"ruột",
        r"đại tràng",
        r"táo bón",
        r"tiêu chảy",
        r"hậu môn",
        r"thực quản",
        r"trào ngược",
        r"polyp",
        r"viêm tá tràng",
    ],

    "Thận – Tiết niệu": [
        r"thận",
        r"tiết niệu",
        r"bàng quang",
        r"sỏi thận",
        r"viêm thận",
        r"\bniệu\b",
    ],

    "Tai Mũi Họng": [
        r"\btai\b",
        r"\bmũi\b",
        r"họng",
        r"thanh quản",
        r"amiđan",
        r"amidan",
        r"xoang",
        r"viêm họng",
        r"điếc",
        r"ù tai",
    ],

    "Mắt": [
        r"mắt",
        r"võng mạc",
        r"thị lực",
        r"giác mạc",
        r"\bmù\b",
        r"thủy tinh thể",
        r"nhãn",
    ],

    "Thần kinh": [
        r"\bnão\b",
        r"thần kinh",
        r"đau nửa đầu",
        r"chóng mặt",
        r"động kinh",
        r"\bliệt\b",
        r"sa sút trí tuệ",
        r"mất trí nhớ",
    ],

    "Cơ xương khớp": [
        r"xương",
        r"khớp",
        r"viêm khớp",
        r"gout",
        r"loãng xương",
        r"thoái hóa",
        r"cột sống",
        r"đĩa đệm",
    ],

    "Da liễu": [
        r"\bda\b",
        r"mụn",
        r"nấm da",
        r"viêm da",
        r"chàm",
        r"vẩy nến",
        r"mề đay",
        r"ghẻ",
        r"zona",
    ],

    "Sản phụ khoa": [
        r"âm đạo",
        r"tử cung",
        r"buồng trứng",
        r"\bvú\b",
        r"kinh nguyệt",
        r"thai",
        r"mang thai",
        r"sản",
    ],

    "Nội tiết": [
        r"tiểu đường",
        r"đái tháo đường",
        r"tuyến giáp",
        r"hormone",
        r"cường giáp",
        r"suy giáp",
        r"tuyến yên",
    ],

    "Truyền nhiễm": [
        r"sốt xuất huyết",
        r"sốt rét",
        r"\bhiv\b",
        r"aids",
        r"thủy đậu",
        r"thương hàn",
        r"viêm gan virus",
        r"ký sinh",
    ],

    "Huyết học": [
        r"thiếu máu",
        r"bạch cầu",
        r"hồng cầu",
        r"đông máu",
        r"bệnh máu",
    ],

    "Nam khoa": [
        r"tuyến tiền liệt",
        r"tinh hoàn",
        r"dương vật",
        r"\bbìu\b",
    ],

    "Tâm thần": [
        r"trầm cảm",
        r"lo âu",
        r"tâm thần",
        r"tự kỷ",
        r"tăng động",
        r"lưỡng cực",
        r"cai nghiện",
    ],

    "Ngoại tổng quát": [
        r"ruột thừa",
        r"thoát vị",
    ],
}

def load_existing_mapping():
    """Load existing mapping"""
    old_map_path = Path(__file__).parent / "department_map_old.json"
    if old_map_path.exists():
        with open(old_map_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def match_department_enhanced(disease_name, existing_map):
    """Enhanced department matching"""
    disease_lower = disease_name.lower().strip()

    # 1. Check specific mappings first
    if disease_name in SPECIFIC_MAPPINGS:
        return SPECIFIC_MAPPINGS[disease_name]

    # 2. Check existing mapping
    if disease_name in existing_map:
        return existing_map[disease_name]

    for k, v in existing_map.items():
        if k.lower().strip() == disease_lower:
            return v

    # 3. Pattern matching with regex
    for dept, patterns in DEPARTMENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, disease_lower):
                return dept

    # 4. Default fallback
    return "Khám tổng quát"

def create_mapping(diseases_file):
    """Create comprehensive mapping"""
    existing_map = load_existing_mapping()
    print(f"Loaded {len(existing_map)} existing mappings")

    with open(diseases_file, 'r', encoding='utf-8') as f:
        diseases = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(diseases)} diseases...")

    mapping = {}
    dept_counts = {}

    for disease in diseases:
        dept = match_department_enhanced(disease, existing_map)
        mapping[disease] = dept
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

    return mapping, dept_counts

def main():
    """Main function"""
    script_dir = Path(__file__).parent
    diseases_file = script_dir / "all_577_diseases.txt"
    output_file = script_dir / "department_map_577.json"

    print("="*80)
    print("ENHANCED DEPARTMENT MAPPING FOR 577 DISEASES")
    print("="*80)

    mapping, counts = create_mapping(diseases_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Mapping saved to: {output_file}")
    print(f"\n📊 Department Distribution:")
    print("-"*80)

    for dept, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(mapping) * 100
        bar = "█" * int(pct / 2)
        print(f"  {dept:30s} {count:3d} ({pct:5.1f}%) {bar}")

    print(f"\n{'='*80}")
    print(f"Total: {len(mapping)} diseases")

    # Show some samples
    print(f"\n📋 Sample mappings (first 30):")
    print("-"*80)
    for i, (disease, dept) in enumerate(list(mapping.items())[:30], 1):
        print(f"  {i:2d}. {disease:50s} → {dept}")

    # Show previously unmapped that are now mapped
    default_count = counts.get("Khám tổng quát", 0)
    print(f"\n📉 Reduced unmapped diseases to: {default_count} ({default_count/len(mapping)*100:.1f}%)")

if __name__ == "__main__":
    main()
