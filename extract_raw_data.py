import json
import os
import matplotlib.pyplot as plt
import numpy
import pandas
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from fancurve import FanCurve


RAW_TXT_FILE = "Picture1.txt"
RAW_JSON_FILE = "fan_data.json"


def main():

    # Check if the JSON file already exists, if not, convert the raw data from the TXT file to JSON format
    if not os.path.exists(RAW_JSON_FILE):
        convert_raw_data_from_txt_to_json(RAW_TXT_FILE)

    # Read the data from the JSON file
    data = read_json_data(RAW_JSON_FILE)

    # Check the data whether it is correct
    check_data(data)


    Blue = "#002ac2"

    colors = {
        "Line #1":  "#d5d200",
        "Line #2": Blue,
        "4 degree": Blue,
        "8 degree":  Blue,  
        "12 degree":  Blue,
    }

    plt.figure(figsize=(12, 8))

    for name, points in data.items():

        curve = FanCurve(
            points=points,
            name=name,
            smooth_factor=0.005,
            resolution=3000,
        )

        print(curve.summary())

        x, y = curve.get_curve()

        plt.plot(
            x,
            y,
            linewidth=2.5,
            color=colors.get(name),
            label=name,
            solid_capstyle="round",
            antialiased=True,
        )

    plt.title("Fan Performance Curves")
    plt.xlabel("Air Flow")
    plt.ylabel("Static Pressure")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()



def check_data(data):
    for key in data:
        x = [point[0] for point in data[key]]
        y = [point[1] for point in data[key]]
        print(f"{key}: {len(data[key])} data points")
        print(f"Median of {key}: {numpy.median(x)}, {numpy.median(y)}")
        print(f"Mean of {key}: {numpy.mean(x)}, {numpy.mean(y)}")
        print(f"Standard Deviation of {key}: {numpy.std(x)}, {numpy.std(y)}")
        print(f"Min of {key}: {numpy.min(x)}, {numpy.min(y)}")
        print(f"Max of {key}: {numpy.max(x)}, {numpy.max(y)}")
        


def read_json_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def convert_raw_data_from_txt_to_json(filename):

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")

    data = {
        "Line #1": [],
        "Line #2": [],
        "4 degree": [],
        "8 degree": [],
        "12 degree": [],
    }

    with open("Picture1.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        print(len(lines), "lines read from the file.")
        
        current_section = None
        for line in lines:
            if line.startswith("Line #1"):
                current_section = "Line #1"
            elif line.startswith("Line #2"):
                current_section = "Line #2"
            elif line.startswith("4 degree"):   
                current_section = "4 degree"
            elif line.startswith("8 degree"):
                current_section = "8 degree"
            elif line.startswith("12 degree"):
                current_section = "12 degree"
                
            print(current_section)

            if current_section is not None and line.strip() and not line.startswith(current_section):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        data[current_section].append((x, y))

                    except ValueError:
                        continue  # Skip lines that don't contain valid numbers

        
    for key in data:
        print(f"{key}: {len(data[key])} data points")

    with open('fan_data.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("Data processing complete. Data saved to 'fan_data.json'.")





if __name__ == "__main__":
    main()