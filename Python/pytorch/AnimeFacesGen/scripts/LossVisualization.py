import matplotlib.pyplot as plt

rfile = open('path/nohup.out')

lossD = []
lossG = []

for line in rfile.readlines():
    if line[line.find('[',1)+1] =='0':
        lossG.append(float(line[line.find('D')+2:line.find('D')+7]))
        lossD.append(float(line[line.find('G')+2:line.find('G')+7]))

fig = plt.figure(1)

lG = plt.plot(range(100),lossG)
lD = plt.plot(range(100),lossD)

plt.xlabel('epoch')
plt.ylabel('Loss-Value')
plt.legend(labels=['LossG','LossD'])

plt.show()