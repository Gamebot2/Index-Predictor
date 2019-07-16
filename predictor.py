import csv

with open('data.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

for i in range(10):
    print(str(i) + ": " + str(data[i]))
