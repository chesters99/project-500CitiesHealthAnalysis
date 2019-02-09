import csv

ori_cities = {}

with open('500_Cities.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    city_header = next(reader)
    for row in reader:
        if row[4] == 'City':
            city_state = '{}, {}'.format(row[3], row[1])
            if not city_state in ori_cities.keys():
                ori_cities[city_state] = []
            ori_cities[city_state].append(row)
print(len(ori_cities.keys()))
for city in ori_cities.keys():
    print(city)
    print(len(ori_cities[city]))
