#!/usr/bin/env python3
"""
Script để chuyển đổi số thành chữ tiếng Việt trong metadata.csv
Giúp model hiểu tốt hơn và tránh lỗi "Character not found"
"""

import os
import re

def number_to_vietnamese_words(number):
    """Chuyển đổi số thành chữ tiếng Việt"""
    # Mapping số cơ bản
    units = ['', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín']
    teens = ['mười', 'mười một', 'mười hai', 'mười ba', 'mười bốn', 'mười lăm', 
             'mười sáu', 'mười bảy', 'mười tám', 'mười chín']
    
    if number == 0:
        return 'không'
    elif number < 10:
        return units[number]
    elif number < 20:
        return teens[number - 10]
    elif number < 100:
        if number % 10 == 0:
            return units[number // 10] + ' mươi'
        elif number % 10 == 1:
            return units[number // 10] + ' mươi mốt'
        elif number % 10 == 5:
            return units[number // 10] + ' mươi lăm'
        else:
            return units[number // 10] + ' mươi ' + units[number % 10]
    elif number < 1000:
        if number % 100 == 0:
            return units[number // 100] + ' trăm'
        else:
            return units[number // 100] + ' trăm ' + number_to_vietnamese_words(number % 100)
    else:
        return str(number)  # Giữ nguyên số lớn

def convert_text_numbers_to_words(text):
    """Chuyển đổi tất cả số trong text thành chữ"""
    def replace_number(match):
        number_str = match.group(0)
        try:
            number = int(number_str)
            return number_to_vietnamese_words(number)
        except ValueError:
            return number_str
    
    # Tìm và thay thế các số
    return re.sub(r'\b\d+\b', replace_number, text)

def process_metadata(input_file, output_file):
    """Xử lý file metadata.csv"""
    print(f"🔄 Đang xử lý file: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file: {input_file}")
        return False
    
    # Đọc file gốc
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 Tổng số dòng: {len(lines)}")
    
    # Xử lý từng dòng
    converted_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                filename = parts[0]
                text = parts[1]
                
                # Chuyển đổi số thành chữ
                converted_text = convert_text_numbers_to_words(text)
                
                # Tạo dòng mới
                new_line = f"{filename}|{converted_text}"
                converted_lines.append(new_line)
                
                # Hiển thị ví dụ
                if i < 5:
                    print(f"  Dòng {i+1}:")
                    print(f"    Gốc: {text}")
                    print(f"    Mới: {converted_text}")
                    print()
            else:
                converted_lines.append(line)
        else:
            converted_lines.append(line)
    
    # Ghi file mới
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in converted_lines:
            f.write(line + '\n')
    
    print(f"✅ Đã lưu file mới: {output_file}")
    print(f"📝 Số dòng đã xử lý: {len(converted_lines)}")
    
    return True

def main():
    """Hàm chính"""
    # Đường dẫn file
    current_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_path, "..", "data", "training_data", "dataset")
    input_file = os.path.join(dataset_path, "metadata.csv")
    output_file = os.path.join(dataset_path, "metadata_converted.csv")
    
    print("🚀 Bắt đầu chuyển đổi số thành chữ tiếng Việt...")
    print(f"📁 Thư mục dataset: {dataset_path}")
    print(f"📄 File gốc: {input_file}")
    print(f"📄 File mới: {output_file}")
    print()
    
    # Xử lý file
    success = process_metadata(input_file, output_file)
    
    if success:
        print()
        print("🎉 Hoàn thành chuyển đổi!")
        print("💡 Bạn có thể:")
        print("   1. Sao lưu file gốc: metadata.csv -> metadata_original.csv")
        print("   2. Đổi tên file mới: metadata_converted.csv -> metadata.csv")
        print("   3. Chạy lại script finetune")
        print()
        print("📋 Lệnh thực hiện:")
        print(f"   cd {dataset_path}")
        print("   copy metadata.csv metadata_original.csv")
        print("   copy metadata_converted.csv metadata.csv")
    else:
        print("❌ Có lỗi xảy ra!")

if __name__ == "__main__":
    main()
