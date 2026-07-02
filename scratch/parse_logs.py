with open("scratch/run_slice_8.log", "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    if "Frame " in line and "Persons:" in line:
        print(line.strip())
