import os
import re

re_data = "out-(sw[a-zA-Z0-9_]+)\.txt"

exp_list = []
all_files = os.listdir("experiments")
data_files_unproc = list(map(lambda one_file: re.findall(re_data, one_file), all_files))
data_files = []
for d in data_files_unproc:
  if d:
    data_files.append(d[0])
data_files = sorted(data_files)
print("Available experiments:")
print(data_files)
print()

for d in data_files:
  print("----------")
  print(f"EXPERIMENT {d}")
  os.system(f"python get_data.py -wi "+d)
  print("----------")
