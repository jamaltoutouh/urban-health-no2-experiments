stations = [4, 8, 11, 16, 17, 18, 24, 27, 35, 36, 38, 39, 40, 47, 48, 49, 50, 54, 55, 56, 57, 58, 59, 60]
metric = 8
table_type = 'both-gaps' #'gap' #, 'time-lower-threshold' # 'gap', 'gap-percentage', 'time-lower-threshold'
whole_data = 'false' # 'false' # true

for station in stations:
    extract = 'python extract-information-from-txt.py {:03d} {:02d}'.format(station, metric)
    transform = 'python transform-data-to-sequence.py {:03d} {:02d}'.format(station, metric)
    #print(extract)
    #print(transform)
    # for season in ['winter', 'spring', 'summer', 'autumn']:
    #     comparing = 'python comparing_by_working-day_season.py {:03d} {:02d} {} >> results.tex'.format(station, metric, season)
    #     print(comparing)
    border = 'python border-effect.py {:03d} {:02d} {} {} >> results.tex'.format(station, metric, table_type, whole_data)
    print(border)


