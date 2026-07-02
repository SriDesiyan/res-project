import sys

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/run_slice_8.log", "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    if "SessionManager" in line or "Customer" in line or "FSM" in line:
        if any(tok in line for tok in ["S2", "S3", "table_2"]):
            print(line.strip())
