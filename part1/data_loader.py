import requests

api = 'https://api.covid19india.org/data.json'
data = requests.get(api)
# with open('data.json', 'wb') as f:
#     f.write(data.content)

def transpose(list_of_dict):
    dict_of_list = {}
    for key in list_of_dict[0]:
        if key == 'date':
            dict_of_list[key] = [dic[key] for dic in list_of_dict]
        else:
            dict_of_list[key] = [int(dic[key]) for dic in list_of_dict]
    return dict_of_list

time_series_data = data.json()['cases_time_series']
time_series_list = transpose(time_series_data)

def get_data():
    return time_series_list