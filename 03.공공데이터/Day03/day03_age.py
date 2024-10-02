import csv

def get_index_of_age_csv():
    f = open('age.csv', encoding='euc_kr')
    data = csv.reader(f)
    header = next(data)
    print('------------------------------------------------------')
    print(' age.csv index ')
    print('------------------------------------------------------')
    for i in range(len(header)):
        print((f'[{i:3}]: {header[i]}'))

    f.close()

get_index_of_age_csv()