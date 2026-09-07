import json
import os

with open('apps/backend/model/efficientnetb0_36class_class_mapping.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)

with open('apps/mobile/lib/core/data/motif_asset_registry.dart', 'r', encoding='utf-8') as f:
    dart_content = f.read()

asset_dir = 'apps/mobile/assets/images/motifs'
existing_files = set(os.listdir(asset_dir))

total_classes = 35
assets_found = 0
assets_missing = 0
registry_matches = 0

print(f"| {'ID':<3} | {'Class Label':<28} | {'Asset Filename':<28} | {'Found':<6} | {'Registry Match':<14} |")
print(f"|{'-'*5}|{'-'*30}|{'-'*30}|{'-'*8}|{'-'*16}|")

for cid in range(35):
    label = mapping[str(cid)]
    key = label.lower().replace('-', '_')
    reg_match = (f"'{key}'" in dart_content) or (f"'{label.lower()}'" in dart_content)
    
    expected_file = None
    for line in dart_content.splitlines():
        if (f"'{key}'" in line or f"'{label.lower()}'" in line) and ':' in line and '.jpg' in line:
            expected_file = line.split(':')[1].strip().replace("'", "").replace(",", "")
            break
            
    if expected_file and expected_file in existing_files:
        found = 'YES'
        assets_found += 1
    else:
        found = 'NO'
        assets_missing += 1
        
    if reg_match:
        registry_matches += 1
        reg_status = 'PASS'
    else:
        reg_status = 'FAIL'
        
    print(f"| {cid:<3} | {label:<28} | {str(expected_file):<28} | {found:<6} | {reg_status:<14} |")

print("==================================================")
print(f"TOTAL BATIK CLASSES: {total_classes}")
print(f"ASSETS FOUND: {assets_found}")
print(f"ASSETS MISSING: {assets_missing}")
reg_result = 'PASS' if registry_matches == total_classes else 'FAIL'
print(f"REGISTRY MATCH: {reg_result}")
print("==================================================")
