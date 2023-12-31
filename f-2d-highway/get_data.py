
data_src_path = None

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-w', '--write', action='store_const', const=True, default=False)
parser.add_argument('-p', '--print', action='store_const', const=True, default=False)
parser.add_argument('-m', '--matplot', action='store_const', const=False, default=True)
parser_output = parser.parse_args()
data_src_path = parser_output.input
should_write_to_file = parser_output.write
should_print = parser_output.print
should_plot = parser_output.matplot



import re
re_anything = ".+?"
re_float = "(-?\d+(?:\.?\d+)?)"

f = open("experiments/out-"+data_src_path+".txt", encoding = "ISO-8859-1")

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
    print()
  def display_minmax(self):
    print(self.label+": minimum and maximum")
    if self.values:
      print(min(self.values))
      print(max(self.values))
    print()
  def file_write_vis_info(self, summary_file):
    summary_file.write(self.label+"\n")
    summary_file.write(", ".join(str(num) for num in self.values))
    summary_file.write("\n\n")
  def file_write_misc_info(self, summary_file):
    summary_file.write(self.label+": minimum and maximum\n")
    if self.values:
      summary_file.write(f"{min(self.values)}\n")
      summary_file.write(f"{max(self.values)}\n")
    summary_file.write("\n")
  

props = []
props.append(Prop("log obs of generated LA on dataset", 'AVG PRED ERR 3. '))
props.append(Prop("discriminator accuracy", 'disc/disc_acc ', True))
props.append(Prop("accuracy of generated HA on dataset", 'ACC 3. '))

is_mean = True
terminated_line = "not terminated"

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
    if re.findall('terminating program', line):
      terminated_line = line
    elif not line:
      break
    







if should_print:
  print("0: ACC")
  print("1: DEC")
  print("2: DEC+LEFT")
  print("3: DEC+RIGHT")
  print()

  for prop in props:
    prop.pretty_print()


 # important - print each time
for prop in props:
  prop.display_minmax()

print(f"timesteps elapsed: {len(props[0].values)-1}")
print(terminated_line)
print()




if should_write_to_file:
  summary = open("experiments/out-"+data_src_path+"-summary.txt", "w")
  for prop in props:
    prop.file_write_vis_info(summary)
  for prop in props:
    prop.file_write_misc_info(summary)



if should_plot:
  print(len(props[1].values))
  print(len(props[2].values))
  import matplotlib.pyplot as plt
  plt.plot(props[1].values)
  plt.plot(props[2].values)
  plt.savefig(f"plots/plot-{data_src_path}.png")

