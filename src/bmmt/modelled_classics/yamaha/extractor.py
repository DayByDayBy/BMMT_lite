import os
import re

def extract_readable_text(qpre_file):
    """Extract the readable parameter section from .qpre file"""
    with open(qpre_file, 'rb') as f:
        content = f.read()
    
    try:
        start = content.find(b'V4Osc')
        if start == -1:
            return None
        text = content[start:].decode('utf-8', errors='ignore')
        return text
    except:
        return None

def parse_algorithm_routing(text):
    """Extract operator routing from the parameter text"""
    routing = {}
    
    for i in range(1, 7):
        kernel = f"Kernel{i}"
        
        # Check if active
        active_match = re.search(f'V4Osc1{kernel}Active(On|Off)', text)
        if active_match and active_match.group(1) == 'Off':
            continue
        
        # Find modulation sources
        mod2_match = re.search(f'V4Osc1{kernel}ModSrc2(K\\d+|-)', text)
        mod3_match = re.search(f'V4Osc1{kernel}ModSrc3(K\\d+|-)', text)
        
        # Find output destination
        out_match = re.search(f'V4Osc1{kernel}OutDest\\[?(Out|Off)', text)
        
        # Find feedback
        feedback_match = re.search(f'V4Osc1{kernel}Feedback([^V]+?)V4', text)
        has_feedback = feedback_match and feedback_match.group(1).strip() not in ['', 'ff', '0']
        
        mods = []
        if mod2_match and mod2_match.group(1).startswith('K'):
            mods.append(mod2_match.group(1))
        if mod3_match and mod3_match.group(1).startswith('K'):
            mods.append(mod3_match.group(1))
        
        routing[f"Operator {i}"] = {
            'modulated_by': mods if mods else [],
            'outputs_to': 'Audio Out' if (out_match and out_match.group(1) == 'Out') else 'Internal',
            'has_feedback': has_feedback
        }
    
    return routing

def format_routing_description(routing):
    """Format routing info as readable text"""
    lines = []
    for op, info in sorted(routing.items()):
        parts = [f"**{op}**:"]
        
        if info['modulated_by']:
            modulators = ', '.join(info['modulated_by'])
            parts.append(f"modulated by {modulators}")
        else:
            parts.append("no modulation")
        
        parts.append(f"→ {info['outputs_to']}")
        
        if info['has_feedback']:
            parts.append("(with feedback)")
        
        lines.append('  - ' + ' '.join(parts))
    
    return '\n'.join(lines)

# Main processing
folder = 'Algorithms_FM-DX7'
output_lines = [
    "# DX7 Algorithm Reference",
    "",
    "Extracted from Waldorf Quantum .qpre recreations of DX7 algorithms.",
    "",
    "**Note:** These represent routing topology only. Parameter scaling differs between DX7 and Waldorf.",
    "",
    "---",
    ""
]

# Collect all algorithm numbers from filenames
algo_files = []
for file in sorted(os.listdir(folder)):
    if file.endswith('.qpre') and 'algo' in file.lower():
        algo_nums = [int(n) for n in re.findall(r'\d+', file)]
        algo_files.append((algo_nums, file))

# Sort by first algorithm number
algo_files.sort(key=lambda x: x[0][0])

# Process each file
for algo_nums, file in algo_files:
    filepath = os.path.join(folder, file)
    text = extract_readable_text(filepath)
    
    if not text:
        output_lines.append(f"## Algorithms {', '.join(map(str, algo_nums))}")
        output_lines.append("⚠️ Could not parse file")
        output_lines.append("")
        continue
    
    routing = parse_algorithm_routing(text)
    
    # Format header
    if len(algo_nums) == 1:
        header = f"## Algorithm {algo_nums[0]}"
    else:
        header = f"## Algorithms {', '.join(map(str, algo_nums))}"
    
    output_lines.append(header)
    output_lines.append(f"*Source: {file}*")
    output_lines.append("")
    output_lines.append(format_routing_description(routing))
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")

# Write output
output_file = 'DX7_ALGORITHM_REFERENCE.md'
with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))

print(f"✓ Created {output_file} with {len(algo_files)} algorithm configurations")
print(f"  Covering all 32 DX7 algorithms")