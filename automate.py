import os
import io
import re
import sys
import time
import json
import argparse
import numpy as np
from io import StringIO

import detection_engine

"""
Sample run:
python automate.py --num_iterations 10 --top_n 25 --min_volume 5000 --data_granularity_minutes 60 --history_to_use 14 --data_dictionary_path 'dictionaries/feature_dict.npy' --output_format 'CLI'
"""

class ArgChecker:
	def __init__(self):
		self.check_arugments()

	def check_arugments(self):
		granularity_constraints_list = [1, 5, 10, 15, 30, 60]
		granularity_constraints_list_string = ''.join(str(value) + "," for value in granularity_constraints_list).strip(",")
		directory_path = str(os.path.dirname(os.path.abspath(__file__)))

		if detection_engine.data_granularity_minutes not in granularity_constraints_list:
			print("You can only choose the following values for 'data_granularity_minutes' argument -> %s\nExiting now..." % granularity_constraints_list_string)
			exit()

		if not os.path.exists(directory_path + f'/stocks/{detection_engine.stock_list}'):
			print("The stocks list file must exist in the stocks directory")
			exit()

		if detection_engine.data_source not in ['binance', 'yahoo_finance']:
			print("Data source must be a valid and supported service.")
			exit()


def get_num_stocks(stock_list):
    with open("stocks/" + stock_list) as stock_file:
        lines = stock_file.readlines()
        lines = [line.strip("\n") for line in lines if line != "\n"]
        return len(lines)


def parse_arguments():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--num_iterations", type=int, default=10, help="The number of iterations to run the detection engine")
    argParser.add_argument("--top_n", type=int, default = 10, help="How many top predictions do you want to print")
    argParser.add_argument("--min_volume", type=int, default = 5000, help="Minimum volume filter. Stocks with average volume of less than this value will be ignored")
    argParser.add_argument("--history_to_use", type=int, default = 7, help="How many bars of 1 hour do you want to use for the anomaly detection model.")
    argParser.add_argument("--data_dictionary_path", type=str, default = "dictionaries/data_dictionary.npy", help="Data dictionary path.")
    argParser.add_argument("--data_granularity_minutes", type=int, default = 15, help="Minute level data granularity that you want to use. Default is 60 minute bars.")
    argParser.add_argument("--volatility_filter", type=float, default = 0.05, help="Stocks with volatility less than this value will be ignored.")
    argParser.add_argument("--stock_list", type=str, default = "stocks.txt", help="What is the name of the file in the stocks directory which contains the stocks you wish to predict.")
    argParser.add_argument("--data_source", type=str, default = "yahoo_finance", help="The name of the data engine to use.")

    args = argParser.parse_args()
    detection_engine.top_n = get_num_stocks(args.stock_list)
    print(detection_engine.top_n)
    detection_engine.min_volume = args.min_volume
    detection_engine.history_to_use = args.history_to_use
    detection_engine.is_load_from_dictionary = 0
    detection_engine.data_dictionary_path = args.data_dictionary_path
    detection_engine.is_save_dictionary = 1
    detection_engine.data_granularity_minutes = args.data_granularity_minutes
    detection_engine.is_test = 0
    detection_engine.future_bars = 0
    detection_engine.volatility_filter = args.volatility_filter
    detection_engine.output_format = "JSON"
    detection_engine.stock_list = args.stock_list
    detection_engine.data_source = args.data_source

    return (args.num_iterations, args.top_n)


def add_data(all_data, json_data):
    for ticker in json_data:
        if ticker["Symbol"] in all_data:
            all_data[ticker["Symbol"]].append(ticker)
        else:
            all_data[ticker["Symbol"]] = [ticker]


def main():
    all_data = {}

    (num_iterations, num_to_print) = parse_arguments()
    arg_checker = ArgChecker()

    old_stdout = sys.stdout
    stdout_buffer = StringIO()
    sys.stdout = stdout_buffer 

    old_stdout.write("Iteraiton 1:\n")
    old_stdout.write("Downloading Stocks:\n")
    supriver = detection_engine.Surpriver()
    supriver.find_anomalies()

    output = stdout_buffer.getvalue()

    filename_regex = re.compile(r"(Results stored successfully in )(.+)\n")
    regex_match = filename_regex.search(output)

    if regex_match:
        file_name = regex_match.group(2)
        json_string = None

        with open(file_name, "r") as f:
            json_string = f.readline().strip()

        os.remove(file_name)

        json_data = json.loads(json_string) 
        add_data(all_data, json_data)

    old_stdout.write("Done\n")

    # Switch from saving the dictionary to loading the dictionary
    detection_engine.is_save_dictionary = 0
    detection_engine.is_load_from_dictionary = 1

    for i in range(num_iterations - 1):
        old_stdout.write("Iteration {}: ".format(i + 2))
        stdout_buffer = StringIO()
        sys.stdout = stdout_buffer

        supriver = detection_engine.Surpriver()
        supriver.find_anomalies()

        output = stdout_buffer.getvalue()

        regex_match = filename_regex.search(output)

        if regex_match:
            filename = regex_match.group(2)
            json_string = None

            with open(filename, "r") as f:
                json_string = f.readline().strip()

            os.remove(filename)

            json_data = json.loads(json_string) 
            add_data(all_data, json_data)

        old_stdout.write("Done\n")

    sys.stdout = old_stdout

    if not os.path.exists("automation_data"):
        os.mkdir("automation_data")
        
    np.save("automation_data/" + detection_engine.data_dictionary_path.split("/")[-1].split(".")[0] + "_automation_results.npy", all_data) # Get the name part (split(".")[0]) of the last part of the path (split("/")[-1])

    # Add a blank line to seperate the iterations from the data
    print()

    averages = []
    for ticker in all_data:
        if len(all_data[ticker]) != 1:
            average = 0
            for datum in all_data[ticker]:
                average += datum["Anomaly Score"]

            average /= len(all_data[ticker])
            averages.append((ticker, average))

    averages.sort(key=lambda item: item[1])

    for i in range(num_to_print):
        print("----------------------")
        print("Symbol: " + averages[i][0])
        print("Avg. Anomaly Score: {:.3f}".format(averages[i][1]))

    print("----------------------")


if __name__ == "__main__":
    main()