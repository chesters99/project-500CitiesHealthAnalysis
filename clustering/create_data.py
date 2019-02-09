import csv
import random
import copy

N_CITIES = [500, 5000, 50000, 500000]

ori_cities = {}
new_cities = None

ori_lat_long = {}
new_lat_long = None

ori_life_exp = {}
new_life_exp = None

new_LM = None

city_header = None
lat_long_header = None
health_ineq_headers = []
LM_header = ['LMA/CZ', 'FIPS', 'County Name', 'Total Population', 'Labor Force']

unique_new_cities = set()


def init():
    global city_header, lat_long_header, health_ineq_header
    with open('500_Cities.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        city_header = next(reader)
        for row in reader:
            if row[4] == 'City':
                city_state = '{}, {}'.format(row[3], row[1])
                if not city_state in ori_cities.keys():
                    ori_cities[city_state] = []
                ori_cities[city_state].append(row)
    with open('uscitiesLatLongFIPS.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        lat_long_header = next(reader)
        for row in reader:
            city_state = '{}, {}'.format(row[1].replace('Saint', 'St.'), row[2])
            ori_lat_long[city_state] = row
    with open('health_ineq_all_online_tables.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i in range(7):
            health_ineq_headers.append(next(reader))
        for row in reader:
            city_state = '{}, {}'.format(row[2].replace('Saint', 'St.'), row[6])
            ori_life_exp[city_state] = row


def sample_cities(n_cities):
    counter = 0
    new_cities = []
    new_lat_long = []
    new_life_exp = []
    new_LM = []
    while counter < n_cities:
        parents = random.sample(sorted(ori_cities.keys()), 2)
        if not (parents[0] in ori_life_exp.keys() and parents[1] in ori_life_exp.keys()):
            continue
        weight = random.random()
        name1 = parents[0].split(',')[0]
        split_idx1 = max(1, round(len(name1) * weight))
        name2 = parents[1].split(',')[0]
        split_idx2 = max(1, round(len(name2) * (1-weight)))
        city_name = (name1[:-split_idx1] + name2[split_idx2:]).title() + str(random.randint(1, 999))
        if city_name in unique_new_cities:
            continue
        unique_new_cities.add(city_name)
        counter += 1
        print(counter)
        FIPS = '{:05d}'.format(counter)
        offspring = []

        state_abbr = parents[0].split(',')[1].strip()
        for p1, p2 in zip(ori_cities[parents[0]], ori_cities[parents[1]]):
            offspring_measure = copy.deepcopy(p1)
            offspring_measure[3] = city_name
            for idx in [12, 13, 14, 17]:
                offspring_measure[idx] = weight * float(p1[idx]) + (1 - weight) * float(p2[idx])
            offspring_measure[21] = counter
            offspring.append(offspring_measure)
        new_cities.append(offspring)
        lat_long_p1 = ori_lat_long[parents[0]]
        lat_long_p2 = ori_lat_long[parents[1]]
        lat_long_offspring = copy.deepcopy(lat_long_p1)
        lat_long_offspring[1] = city_name
        #arbitrary number for fips
        lat_long_offspring[5] = FIPS
        for idx in [7, 8]:
            lat_long_offspring[idx] = weight * float(lat_long_p1[idx]) + (1 - weight) * float(lat_long_p2[idx])
        new_lat_long.append(lat_long_offspring)
        life_exp_p1 = ori_life_exp[parents[0]]
        life_exp_p2 = ori_life_exp[parents[1]]
        life_exp_offspring = copy.deepcopy(life_exp_p1)
        #same number as above for cz
        life_exp_offspring[1] = FIPS
        life_exp_offspring[2] = city_name
        #same number as above for fips
        life_exp_offspring[4] = FIPS
        for idx in [8, 52]:
            if life_exp_p1[idx] == '':
                continue
            life_exp_offspring[idx] = weight * float(life_exp_p1[idx].replace(',', '')) + \
                                      (1 - weight) * float(life_exp_p2[idx].replace(',', ''))
        new_life_exp.append(life_exp_offspring)
        LM_offspring = []
        #first row is 'LMA/CZ',
        LM_offspring.append(FIPS)
        #second row is county 'FIPS' (can use same id as for LMA, as long as unique within this table, but zeropad!)
        LM_offspring.append(FIPS)
        LM_offspring.append('{}, {}'.format(city_name, state_abbr))#parents[0] + '_' + parents[1])
        #arbitrary number for the remaining two
        LM_offspring.append(999)
        LM_offspring.append(99)
        new_LM.append(LM_offspring)

    with open('new_{}_cities.csv'.format(n_cities), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(city_header)
        for city in new_cities:
            writer.writerows(city)
    with open('new_{}_lat_lon.csv'.format(n_cities), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(lat_long_header)
        writer.writerows(new_lat_long)
    with open('new_{}_health_ineq_all_online_tables.csv'.format(n_cities), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(health_ineq_headers)
        writer.writerows(new_life_exp)
    with open('new_{}_1990LMAascii.csv'.format(n_cities), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(LM_header)
        writer.writerows(new_LM)


if __name__ == '__main__':
    init()
    random.seed(123)
    for n_cities in N_CITIES:
        print(n_cities)
        sample_cities(n_cities)
