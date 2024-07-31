# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
import sys
import argparse

def read_numbers_from_file(file_path, col_start, col_end):
    try:
        numbers = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            start_index = None
            for i, line in enumerate(lines):
                if "step    1/10" in line:
                    start_index = i
                    break

            if start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                return None

            # Read 10 rows starting from the identified start row
            for line in lines[start_index:start_index + 10]:
                # Extracting the specified columns
                number = float(line[col_start:col_end].strip())
                numbers.append(number)
        return numbers
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

def compare_numbers(read_values, fixed_values, percent_accuracy):
    for i in range(len(read_values)):
        read_value = read_values[i]
        fixed_value = fixed_values[i]
        percent_difference = ((read_value - fixed_value) / fixed_value) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        if abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            return 1
    print("Success: All values are within the allowed accuracy.")
    return 0

def main():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=True, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=int, required=True, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=int, required=True, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=float, required=True, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_values = read_numbers_from_file(args.file, args.col_start, args.col_end)
    if read_values is None:
        return 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_values = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_values, fixed_values, args.percent_accuracy)
    return result

if __name__ == "__main__":
    sys.exit(main())
