import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def get_lambda(half_life):
    return np.log(2) / half_life

def get_half_life(lbda):
    return np.log(2) / lbda

def linear_function(m, x, c):
    return m * x + c

def decay_function(lbda, x, S0):
    return S0 * np.exp(-lbda * x)

def get_lines_from_file(filename):
    print(f"Loading lines from file: \n  {filename}")
    with open(filename) as f:
        lines = f.readlines()
    return lines

def extract_singles_values_array_from_lines(lines):
    """
    This is a poorly written function that loops over each of the lines of the log file.
    The function filters the lines, looking for array depth information, given by "{" and "}".
    There are a few lines begining with "(" or "Time frame" or are empty that are ignored.
    """
    print("\nExtracting singles values array from lines...")
    line_num = 0
    num_lines = len(lines)
    singles_values = [[]]
    i = 0
    for l in lines:
        line_num += 1
        if line_num % 100 == 0:
            print(f"  Processing {line_num}/{num_lines} ({round(line_num / num_lines * 1000) / 10}%) lines...",
                  end="\r") # funky way to give percentages with decimal, e.g. 12.3%

        # Remove unneeded lines
        if len(l) <= 1 or l.startswith("(") or l.startswith("Time frame"):
            continue
            
        if l.count("{") == 2:
            # Increasing list depth
            i += 1
            singles_values.append([])
            
        if l.startswith(", "):
            # first 2 characters of the line are ", ", which we remove here.
            l = l[2::] 
            
        # remove any "{", "}", "\n" and split the remaining string into a list by "," as the delimiter.
        # Do not edit the original l
        tmp = l.replace("{", "").replace("}", "").replace("\n", "").split(",")
        
        
        if len(tmp) > 1:
            # Expect a list of some length, otherwise it may be an empty, which is discarded here.
            tmp = [float(i) for i in tmp] # Convert strings into float values.
            singles_values[i].append(tmp) 
            
    print(f"  Processed {line_num}/{num_lines} lines.")
    print("Converting to array...\n")
    return np.array(singles_values) # Ideally this whole function uses arrays rather than lists, this conversion is expensive.

# # Main script

# This next line is the main argument for the script. This should be a GE list mode file. 
# For development, I have hard coded this line.

lm_file = "/Users/roberttwyman/bin/Experiments/Phantoms/GE_data/OriginalFiles/torso-phantom-DMI3ring-RDF9/list/LIST0000.blfun"

# Output the print_GE_singles_values into a log file. No need to rerun once it has been run once.

log_file = "print_GE_singles_values.log"
if not os.path.isfile(log_file):
    print(f"Running print_GE_singles_values command! Saving into {log_file}")
    os.system(f"print_GE_singles_values {lm_file} > {log_file}")
else:
    print(f"{log_file} already exists.")

# Load the log file in text and imediately load into a function to extract the array

singles_values = extract_singles_values_array_from_lines(get_lines_from_file(log_file))

# %

seconds = np.array([0] * len(singles_values)) # Assume each array corresponds 10 one second

singles_per_second = np.array([0] * len(singles_values))
for i in range(len(singles_values)):
    seconds[i] = i
    singles_per_second[i] = np.sum(singles_values[i])

# %%
# log( S0 exp(-2lambda t) = log(S0) -2lambda t )

lbda, S0 = np.polyfit(seconds, np.log(singles_per_second), 1)
lbda = -lbda
S0 = np.exp(S0)
measured_half_life = get_half_life(lbda)

F18_half_life = 6586.2
F18_lbda = get_lambda(F18_half_life)

# lin_fit = [linear_function(lbda, si, np.log(S0)) for si in seconds]

exponential_fit = [decay_function(lbda, t, S0) for t in seconds]
F18_singles = [decay_function(F18_lbda, t, S0) for t in seconds]

plt.figure()
plt.plot(singles_per_second)
plt.plot(exponential_fit)
plt.plot(F18_singles)
plt.title(f"fit half-life = {round(measured_half_life*10)/10} seconds")
plt.legend(["singles_per_second", "exponential fit to measured data", "F18 decay (using correct t1/2)"])
plt.ylabel("Single Events per Second")
plt.xlabel("Seconds (s)")
plt.show()

# 
# plt.figure()
# plt.plot(np.log(singles_per_second))
# plt.plot(lin_fit)
# plt.legend(["log[singles_per_second]", "linear_fit"])
# plt.title(f"lambda = {1 / lbda}, log[S0] = {S0}")
# plt.ylabel("log[Single Events per Second]")
# plt.xlabel("Seconds (s)")
# plt.show()

print(f"sum(singles_per_second) / sum(exponential_fit) = {np.sum(singles_per_second) / np.sum(exponential_fit)}\n",
      f"sum(F18_singles) / sum(exponential_fit) = {np.sum(F18_singles) / np.sum(exponential_fit)}\n",
      f"measured half_life = {measured_half_life}\n"
      f"F18_half_life = {F18_half_life}")

# %

plt.figure()
plt.imshow(np.sum(singles_values, axis=0))
plt.show()

# %

plt.figure()
plt.plot(np.sum(singles_values, axis=(0,1)))
plt.plot(np.flip(np.sum(singles_values, axis=(0,1))))
plt.title("detectors")
plt.show()

plt.figure()
plt.plot(np.sum(singles_values, axis=(0,2)))
plt.plot(np.flip(np.sum(singles_values, axis=(0,2))))
plt.title("rings")
plt.show()