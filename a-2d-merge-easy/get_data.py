
data_src_path = None

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-w', '--write', action='store_const', const=True, default=False)
parser_output = parser.parse_args()
data_src_path = parser_output.input
should_write_to_file = parser_output.write



import re
re_anything = ".+?"
re_float = "(-?\d+\.\d+)"

f = open(data_src_path+".txt")

class Prop:
  def __init__(self, label, re_selector, only_if_mean=False):
    self.label = label
    self.re_selector = re_selector + re_anything + re_float
    self.values = []
    self.only_if_mean = only_if_mean
  def check_for_value(self, line, is_mean):
    res = re.findall(self.re_selector, line)
    if res and (not self.only_if_mean or is_mean):
      self.values.append(float(res[0]))
      return True
    else:
      return False
  def pretty_print(self):
    print(self.label)
    for elem in self.values:
      print(elem)
  def display_minmax(self):
    print(self.label+": minimum and maximum")
    print(min(self.values))
    print(max(self.values))
  def file_write(self, summary_file):
    summary_file.write(self.label+"\n")
    summary_file.write(", ".join(str(num) for num in self.values))
    summary_file.write("\n")
  

props = []
props.append(Prop("discriminator accuracy", 'disc/disc_acc ', True))
props.append(Prop("log obs of generated LA on dataset", 'AVG PRED ERR 3. '))
props.append(Prop("accuracy of generated HA on dataset", 'ACC 3. '))

is_mean = True
while True:
  line = f.readline()
  if re.findall('raw\/', line):
    is_mean = False
  elif re.findall('mean\/', line):
    is_mean = True
  else:
    for prop in props:
      if prop.check_for_value(line, is_mean):
        break
    if not line:
      break
    








print("0: ACC")
print("1: DEC")
print("2: DEC+LEFT")
print("3: DEC+RIGHT")
print()

for prop in props:
  prop.pretty_print()
for prop in props:
  prop.display_minmax()
if should_write_to_file:
  summary = open("summaries/summary_"+data_src_path+".txt", "w")
  for prop in props:
    prop.file_write(summary)

print()
print(f"timesteps elapsed: {len(props[0].values)-1}")
print()