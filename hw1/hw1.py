from typing import Callable, Dict, List, Tuple
import math
import pandas as pd
import matplotlib.pyplot as plt
import texttable as tt

def print_section_1_2_stats(data: pd.DataFrame):
    print('---Basic Data Statistics---')
    print('Entries:', len(data))
    print()
    print_basic_stats(data, 'Rented Bike Count', {
                      'mean': True,  'stdev': True, 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [25, 75]})
    print_basic_stats(data, 'Temperature(C)', {
                      'mean': True,  'stdev': True, 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [25, 75]})
    print_basic_stats(data, 'Humidity(%)', {'mean': True,  'stdev': True, 'mode': True,
                                            'median': True, 'min': True, 'max': True, 'percentiles': [25, 75]})
    print_basic_stats(data, 'Wind speed (m/s)', {'mean': True,  'stdev': True,
                                                 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [25, 75]})
    print_basic_stats(data, 'Visibility (10m)', {
                      'mean': True,  'stdev': True, 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [25, 75]})
    print_basic_stats(data, 'Dew point temperature(C)', {
                      'mean': True,  'stdev': True, 'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [25, 75]})
    print_basic_stats(data, 'Solar Radiation (MJ/m2)', {'mean': True,  'stdev': True,
                                                        'mode': True, 'median': True, 'min': True, 'max': True, 'percentiles': [25, 75]})
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


# finds outliers in the data, sorted by the specified parameter
def print_section_1_4_outliers(data: pd.DataFrame, outlier_src: str, sort_by: str):
    # finding outliers
    temps_dict = sort_by_categories_to_dict(data, outlier_src, sort_by)
    summary = {'Total Outliers': 0, 'Max Outliers In a Day': 0,
               'Number of Days w/ Outliers': 0, 'Days w/ Outliers': [], 'Number Days Examined': 0}
    for date in temps_dict.keys():
        summary_initial = find_outliers(
            pd.Series(temps_dict[date]), is_outlier)
        if summary['Max Outliers In a Day'] < summary_initial['Outliers Count']:
            summary['Max Outliers In a Day'] = summary_initial['Outliers Count']
            summary['Day w/ Max Outliers'] = date
        if summary_initial['Outliers Count'] > 0:
            summary['Number of Days w/ Outliers'] += 1
            summary['Days w/ Outliers'].append(date)
        summary['Total Outliers'] += summary_initial['Outliers Count']
        summary['Number Days Examined'] += 1

    table = tt.Texttable()
    table.add_rows([['', 'Count'],
                    ['Total Outliers:', summary['Total Outliers']],
                    ['Max Outliers In a Day:', summary['Max Outliers In a Day']],
                    ['Number of Days w/ Outliers:', summary['Number of Days w/ Outliers']],
                    ['Number Days Examined:', summary['Number Days Examined']]])
    table.set_cols_align(['l','c'])
    print(table.draw())

# prints stats on attribute in data
def print_basic_stats(data: pd.DataFrame, attribute: str, stats_to_print: Dict[str, None]):
    print_header(attribute)
    is_numeric = data.dtypes[attribute] == 'int64' or data.dtypes[attribute] == 'float64'
    table = tt.Texttable()
    table.add_row(['Statistic', 'Value'])
    if is_numeric: # print numeric stats if numeric
        if stats_to_print['mean']:
            table.add_row(["Mean:", data[attribute].mean()])
        if stats_to_print['stdev']:
            table.add_row(["Std. Dev.:", data[attribute].std()])
        if stats_to_print['mode']:
            table.add_row(["Mode:", data[attribute].mode()[0]])
        if stats_to_print['median']:
            table.add_row(["Median:", data[attribute].median()])
        if stats_to_print['min']:
            table.add_row(["Min:", data[attribute].min()])
        if stats_to_print['max']:
            table.add_row(["Max:", data[attribute].max()])
        for percentile in stats_to_print['percentiles']:
            table.add_row([str(percentile) + "th Percentile:", data.quantile(percentile / 100, 0, numeric_only=True)[attribute]])
    else: # count number of each value if non-numeric
        dict = {}
        for data in data[attribute]:
            if data in dict:
                dict[data] += 1
            else:
                dict[data] = 0
        for key in dict.keys():
            table.add_row([str(key), dict[key]])
    print(table.draw())
    print()

def print_header(title: str):
    print("---", title, "---")

# sums the values in col_to_sum for each category_col
def sum_by_categories_to_dict(data: pd.Series, col_to_sum: str, category_col: str) -> Dict[str, float]:
    dict = {}
    for i in range(0, len(data)):
        key = data[category_col][i]
        if key in dict: # add sum if category is already in dictionary
            dict[key] += data[col_to_sum][i]
        else: # initialize new entry in dictionary if not in there 
            dict[key] = data[col_to_sum][i]
    return dict

# same as sort_by_categories_to_dict, but also includes the index of the entry
def sort_by_categories_to_dict_w_index(data: pd.DataFrame, col_to_sort: str, category_col: str) -> Dict[str, List[Tuple[None, int]]]:
    dict = {}
    for i in range(0, len(data)):
        key = data[category_col][i]
        if key in dict:
            dict[key].append((data[col_to_sort][i], i))
        else:
            dict[key] = [(data[col_to_sort][i], i)]
    return dict

# sorts entries in col_to_sort into categories based on category_col
def sort_by_categories_to_dict(data: pd.DataFrame, col_to_sort: str, category_col: str) -> Dict[str, List[None]]:
    dict = {}
    for i in range(0, len(data)):
        key = data[category_col][i]
        if key in dict: # add to list if key is already in dictionary
            dict[key].append(data[col_to_sort][i])
        else: # create new list if key not in dictionary
            dict[key] = [data[col_to_sort][i]]
    return dict


# def convert_date_dict_to_month_day(dict):
#     converted = {}
#     for key in dict.keys():
#         date = pd.Timestamp(month=key.month, day=key.day,
#                             year=2000, nanosecond=None)
#         if date in dict:
#             converted[date] += dict[key]
#         else:
#             converted[date] = dict[key]
#     return converted

# only keeps the keys for which should_keep returns true for value
def keep_keys(dict: Dict[None, None], should_keep: Callable[[None, None], bool],  value: None) -> Dict[None, None]:
    new_dict = {}
    for entry in dict.keys():
        if should_keep(entry, value):
            new_dict[entry] = dict[entry]
    return new_dict

# imports data from CSV into pd.DataFrame
def data() -> pd.DataFrame:
    # import data from CSV
    data = pd.read_csv("altered_seoulbokedata_train.csv", true_values=[
                       'Holiday', 'Yes'], false_values=['No Holiday', 'No'], parse_dates=True)
    
    # change the dates to timestamps to ease processing
    data['Date'] = data.astype({'Date': 'datetime64'})['Date']
    
    # some columns have special characters (like degree sign and infinity sign)- fix this to make it easier to access them
    cols = [col for col in data]
    cols[2], cols[6] =  'Temperature(C)', 'Dew point temperature(C)'
    data.columns = cols
    return data

# calculate humidity via the equation found here 
def humidity(dp: float, temp: float) -> float:
    c, b = 243.04, 17.625
    return 100 * math.e ** ( (c*b*(dp-temp)) / ((temp + c) * (dp + c)) )

# finds outliers in data using function is_outlier and returns a summary in form of dictionary
def find_outliers(data: pd.Series, is_outlier: Callable[[pd.Series, None], bool]) -> Dict[str, None]:
    summary = {}
    summary["Mean"] = data.mean()
    summary["St. Dev."] = data.std()
    summary["Outliers"] = [value for value in data if is_outlier(data, value)]
    summary["Outliers Count"] = len(summary['Outliers'])
    if summary["Outliers Count"] > 0:
        summary["Max Outlier"] = max(summary["Outliers"])
        summary["Min Outlier"] = min(summary["Outliers"])
        summary["Avg. Outlier"] = sum(summary["Outliers"]) / len(summary['Outliers'])
    return summary

# removes outliers from data using is_outlier function and returns new series and number removed outliers
def remove_outliers(data: pd.Series, is_outlier: Callable[[pd.Series, None], bool]) -> Tuple[pd.Series,int]:
    new_series = pd.Series(data=[val for val in data if not is_outlier(data, val)], name=data.name)
    return new_series, len(data) - len(new_series)
        
# uses 1.5xIQR to determine if value is an outlier in data
def is_outlier(data: pd.Series, value: None) -> bool:
    sensitivity = 1.5
    quartile_one, quartile_three = data.quantile(0.25), data.quantile(0.75)
    iqr = quartile_three - quartile_one
    lower_lim, upper_lim = quartile_one - iqr * \
        sensitivity, quartile_three + iqr * sensitivity
    return value < lower_lim or value > upper_lim

# other graphs that I ended up not using but wanted to save the code for so I could use it later if needed
def other_graphs():
    # Bikes Rented vs. Rainfall
    x, y = 'Rainfall(mm)', 'Rented Bike Count'
    plt.scatter(x=data[x], y=data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Bikes Rented vs. Rainfall')

    # Bikes rented vs. Snowfall
    x, y = 'Snowfall (cm)', 'Rented Bike Count'
    plt.scatter(x=data[x], y=data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Bikes Rented vs. Snowfall')

    # Bikes Rented vs. Solar Radiation
    x, y = 'Solar Radiation (MJ/m2)', 'Rented Bike Count'
    plt.scatter(x=data[x], y=data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Bikes Rented vs. Solar Radiation')

    # Bikes Rented vs. Wind Speed
    x, y = 'Wind speed (m/s)', 'Rented Bike Count'
    plt.scatter(x=data[x], y=data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Bikes Rented vs. Wind Speed')

    # Bikes Rented vs. Humidity
    x, y = 'Humidity(%)', 'Rented Bike Count'
    plt.scatter(x=data[x], y=data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Bikes Rented vs. Humidity')

    # Bikes Rented vs. Visibility
    x, y = 'Visibility (10m)', 'Rented Bike Count'
    plt.scatter(x=data[x], y=data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Bikes Rented vs. Visibility')
