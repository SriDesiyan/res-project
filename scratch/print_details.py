import sys

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/run_slice_8.log", "r", encoding="utf-8") as f:
    lines = f.readlines()

in_range = False
for line in lines:
    if "Frame 74950" in line:
        in_range = True
    if in_range:
        print(line.strip())
    if "Frame 75050" in line:
        in_range = False
