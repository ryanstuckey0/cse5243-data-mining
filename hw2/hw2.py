from typing import Callable, Dict, List, Tuple
import math
import pandas as pd
import matplotlib.pyplot as plt
import texttable as tt


def hw1_transformation(data: pd.DataFrame):
    data_original_size = len(data)
    # find temperature outliers
    day_to_temp_dict = sort_by_categories_to_dict_w_index(
        data, 'Temperature(C)', 'Date')
    temp_outlier_indices = []
    for date in day_to_temp_dict:
        for t in day_to_temp_dict[date]:
            if is_outlier(pd.Series([temp[0] for temp in day_to_temp_dict[date]]), t[0]):
                temp_outlier_indices.append(t[1])

    # calculate dew point outliers
    day_to_dew_point_dict = sort_by_categories_to_dict_w_index(
        data, 'Dew point temperature(C)', 'Date')
    dew_point_outlier_indices = []
    for date in day_to_dew_point_dict:
        for t in day_to_dew_point_dict[date]:
            if is_outlier(pd.Series([temp[0] for temp in day_to_dew_point_dict[date]]), t[0]):
                dew_point_outlier_indices.append(t[1])

    humidity_elims, temp_elims = 0, 0
    # drop any data with errors or outliers
    for row in data.itertuples():
        # drop rows with outlier temps
        if row[0] in temp_outlier_indices:
            data.drop(row[0], inplace=True)
            temp_elims += 1
            continue

        # eliminate humidities based on calculation, but skip check if a) dew point is greater than temperature, or b) dew point is an outlier
        if row[7] > row[3] or row[0] in dew_point_outlier_indices:
            continue
        humidity_rec, humidity_calc = row[4], humidity(row[7], row[3])
        # eliminate row if: a) humidity <= 0m b) recorded humidity not in (calculated humidity +- 5)
        if row[4] <= 0 or humidity_rec < humidity_calc - 5 or humidity_rec > humidity_calc + 5:
            data.drop(row[0], inplace=True)
            humidity_elims += 1
            continue

    table = tt.Texttable()
    table.add_row(['Elimination Cause', 'Count'])
    table.add_row(['Temperature Outlier', temp_elims])
    table.add_row(['Humidity Error', humidity_elims])
    table.add_row(['Total Eliminations', data_original_size - len(data)])
    print(table.draw())

    target_class = remove_cols(data)
    return target_class

def remove_cols(data: pd.DataFrame):
    target_class = data['IsitDay']
    data_to_remove = ['Visibility (10m)', 'Dew point temperature(C)', 'Rainfall(mm)', 'Snowfall (cm)', 'Seasons', 'Holiday', 'Functioning Day', 'Date', 'IsitDay']
    for col in data_to_remove:
        data.drop(col, axis=1, inplace=True)
    return target_class


# prints stats on attribute in data
def print_basic_stats(data: str, attribute: str, stats_to_print: Dict[str, None]):
    print_header(attribute)
    is_numeric = data.dtypes[attribute] == 'int64' or data.dtypes[attribute] == 'float64'
    table = tt.Texttable()
    table.add_row(['Statistic', 'Value'])
    if is_numeric:  # print numeric stats if numeric
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
            table.add_row([str(percentile) + "th Percentile:",
                           data.quantile(percentile / 100, 0, True)[attribute]])
    else:  # count number of each value if non-numeric
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
        if key in dict:  # add sum if category is already in dictionary
            dict[key] += data[col_to_sum][i]
        else:  # initialize new entry in dictionary if not in there
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
        if key in dict:  # add to list if key is already in dictionary
            dict[key].append(data[col_to_sort][i])
        else:  # create new list if key not in dictionary
            dict[key] = [data[col_to_sort][i]]
    return dict


def convert_date_dict_to_month_day(dict):
    converted = {}
    for key in dict.keys():
        date = pd.Timestamp(month=key.month, day=key.day,
                            year=2000, nanosecond=None)
        if date in dict:
            converted[date] += dict[key]
        else:
            converted[date] = dict[key]
    return converted


# only keeps the keys for which should_keep returns true for value
def keep_keys(dict: Dict[None, None], should_keep: Callable[[None, None], bool],  value: None) -> Dict[None, None]:
    new_dict = {}
    for entry in dict.keys():
        if should_keep(entry, value):
            new_dict[entry] = dict[entry]
    return new_dict


# imports data from CSV into pd.DataFrame
def data(file: str) -> pd.DataFrame:
    # import data from CSV
    data = pd.read_csv(file, true_values=[
                       'Holiday', 'Yes'], false_values=['No Holiday', 'No'], parse_dates=True)

    # change the dates to timestamps to ease processing
    data['Date'] = data.astype({'Date': 'datetime64'})['Date']

    # some columns have special characters (like degree sign and infinity sign)- fix this to make it easier to access them
    cols = [col for col in data]
    cols[2], cols[6] = 'Temperature(C)', 'Dew point temperature(C)'
    data.columns = cols
    return data


# calculate humidity via the equation found here
def humidity(dp: float, temp: float) -> float:
    c, b = 243.04, 17.625
    return 100 * math.e ** ((c*b*(dp-temp)) / ((temp + c) * (dp + c)))


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
        summary["Avg. Outlier"] = sum(
            summary["Outliers"]) / len(summary['Outliers'])
    return summary


# removes outliers from data using is_outlier function and returns new series and number removed outliers
def remove_outliers(data: pd.Series, is_outlier: Callable[[pd.Series, None], bool]) -> Tuple[pd.Series, int]:
    new_series = pd.Series(
        data=[val for val in data if not is_outlier(data, val)], name=data.name)
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
