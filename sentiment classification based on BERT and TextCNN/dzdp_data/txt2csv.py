import csv
import pandas as pd


def txt2csv(csvdir, txtdir):
    csvFile = open(csvdir, 'w', newline='', encoding='ANSI')
    writer = csv.writer(csvFile)
    csvRow = []

    f = open(txtdir, 'r', encoding='utf-8')
    for line in f:
        csvRow = line.split()
        writer.writerow(csvRow)
    f.close()

    writer.writerow(csvRow)
    csvFile.close()


if __name__ == '__main__':
    txt2csv('original.csv', 'shuffle.txt')

    names = ['label', 'content']
    names_ = ['content']
    df = pd.read_csv('original.csv', encoding='ANSI', names=names)
    print(df)
