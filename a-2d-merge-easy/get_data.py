
import re

re_float = "(\d+\.\d+)"

f = open("out3.txt")
is_mean = True

# class Prop:
#   Prop(name, re_selector):
  
props_list = []
score_list = []
acc_list = []

while True:
  line = f.readline()
  if re.findall('raw\/', line):
    is_mean = False
  elif re.findall('mean\/', line):
    is_mean = True
  elif res := re.findall('disc/disc_acc .+'+re_float, line):
    if is_mean:
      props_list.append(float(res[0]))
  elif res := re.findall('AVG PRED ERR -'+re_float, line):
    score_list.append(-float(res[0]))
  elif res := re.findall('ACC '+re_float, line):
    acc_list.append(float(res[0]))
  elif not line:
    break
    









print("0: ACC")
print("1: DEC")
print("2: DEC+LEFT")
print("3: DEC+RIGHT")

print("discriminator accuracy")
for elem in props_list:
  print(elem)

print("log observation likelihood")
for elem in score_list:
  print(elem)

print("generator accuracy on dataset")
for elem in acc_list:
  print(elem)


