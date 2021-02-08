import pandas as pd
import matplotlib.pyplot as plt

def print_basic_stats(data, attribute, stats_to_print):
    print_header(attribute)
    is_numeric = data.dtypes[attribute] == 'int64' or data.dtypes[attribute] == 'float64'
    if is_numeric:
        if stats_to_print['mean']:
            print("Mean:", data[attribute].mean())
        if stats_to_print['stdev']:
            print("Std. Dev.:", data[attribute].std())
        if stats_to_print['mode']:
            print("Mode:", data[attribute].mode()[0])
        if stats_to_print['median']:
            print("Median:", data[attribute].median())
        if stats_to_print['min']:
            print("Min:", data[attribute].min())
        if stats_to_print['max']:
            print("Max:", data[attribute].max())
        for percentile in stats_to_print['percentiles']:
            print(percentile, "th Percentile:",
                  data.quantile(percentile / 100, 0, True)[attribute], sep="")
    else:
        dict = {}
        for data in data[attribute]:
            if data in dict:
                dict[data] += 1
            else:
                dict[data] = 0
        for key in dict.keys():
            print(key, dict[key], sep=": ")
    print()


def print_section_1_2(data):
    print_basic_stats(data, 'Rented Bike Count', {
                      'mean': True,  'stdev': True, 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [10, 90]})
    print_basic_stats(data, 'Temperature(C)', {
                      'mean': True,  'stdev': True, 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [10, 90]})
    print_basic_stats(data, 'Humidity(%)', {'mean': True,  'stdev': True, 'mode': True,
                                            'median': True, 'min': True, 'max': True, 'percentiles': [10, 90]})
    print_basic_stats(data, 'Wind speed (m/s)', {'mean': True,  'stdev': True,
                                                 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [10, 90]})
    print_basic_stats(data, 'Visibility (10m)', {
                      'mean': True,  'stdev': True, 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [10, 90]})
    print_basic_stats(data, 'Dew point temperature(C)', {
                      'mean': True,  'stdev': True, 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [10, 90]})
    print_basic_stats(data, 'Solar Radiation (MJ/m2)', {'mean': True,  'stdev': True,
                                                        'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [10, 90]})
    print_basic_stats(data, 'Rainfall(mm)', {
                      'mean': True,  'stdev': True, 'mode': False, 'median': False, 'min': False, 'max': True, 'percentiles': []})
    print_basic_stats(data, 'Snowfall (cm)', {
                      'mean': True,  'stdev': True, 'mode': False, 'median': False, 'min': False, 'max': True, 'percentiles': []})
    print_basic_stats(data, 'Seasons', {'mean': True,  'stdev': True, 'mode': True,
                                        'median': True, 'min': True, 'max': True, 'percentiles': []})
    print_basic_stats(data, 'Holiday', {'mean': True,  'stdev': True, 'mode': True,
                                        'median': True, 'min': True, 'max': True, 'percentiles': []})
    print_basic_stats(data, 'Functioning Day', {
                      'mean': True,  'stdev': True, 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': []})
    print_basic_stats(data, 'IsitDay', {'mean': True,  'stdev': True, 'mode': True,
                                        'median': True, 'min': True, 'max': True, 'percentiles': []})

def print_header(title):
    print("---", title, "---")

def plot_bike_cnt_vs_season(data):
    plt.bar(x=data['Rented Bike Count'],)
    ...

def count_in_categories(data, col_to_sum, lbl_src_col):
    dict = {}
    for i in range(0,len(data)):
        key = data[lbl_src_col][i]
        if key in dict:
            dict[key] += data[col_to_sum][i]
        else:
            dict[key] = 0
    return dict

def data():
    data = pd.read_csv("altered_seoulbokedata_train.csv", true_values=['Holiday', 'Yes'], false_values=['No Holiday', 'No'], parse_dates=True)
    data['Date'] = data.astype({'Date': 'datetime64'})['Date']
    return data