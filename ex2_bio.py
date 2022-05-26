# Asaf Levi 205543317, Hadassa Danesh 3225677041
import random
import numpy as np
import sys

filename = sys.argv[1]
with open(filename, "r") as file:  # file that contains the input of signs and given numbers.
    input_data = file.read().splitlines()

# Global variables.
method = int(sys.argv[2])  # the method that's been chosen when running the program.
METHOD = ""
COORDINATES = {}  # The i - 1,j - 1 coordinates and the value of the given numbers
MATRIX_SIZE = int(input_data[0])  # The size of the matrix
NUMBER_OF_GIVEN_DIGITS = int(input_data[1])  # number of given digits (0 if the matrix is given empty)
NUMBER_OF_GRATER_THAN_SIGNS = int(input_data[NUMBER_OF_GIVEN_DIGITS + 2])  # The number of “greater than” signs
SIGN_COORDINATES = []  # contains all the given signs locations
GRID_LIST = []  # list contains all the grids
NUM_OF_GRIDS = 0  # number of grids to work with.


# Handles with the inputs, arranging them in organized structures. happens once at the beginning of the program.
def start_up():
    global METHOD
    global NUM_OF_GRIDS
    if method == 1:
        METHOD = "Regular"
    if method == 2:
        METHOD = "Darvin"
    if method == 3:
        METHOD = "Lamarck"
    for m in range(0, NUMBER_OF_GIVEN_DIGITS):  # initial coordinates of the given numbers
        given_digit = input_data[2 + m]
        row = int(given_digit[0]) - 1
        col = int(given_digit[2]) - 1
        val = int(given_digit[4])
        COORDINATES[(row, col)] = val
    for m in range(0, NUMBER_OF_GRATER_THAN_SIGNS):  # initial coordinates of the given signs
        given_sign = input_data[3 + NUMBER_OF_GIVEN_DIGITS + m]
        row1 = int(given_sign[0])
        col1 = int(given_sign[2])
        row2 = int(given_sign[4])
        col2 = int(given_sign[6])
        s = ((row1, col1), (row2, col2))
        SIGN_COORDINATES.append(s)
    # Number of grids to work with, the bigger the matrix, we want to store more data of grids.
    if MATRIX_SIZE == 5:
        NUM_OF_GRIDS = 100
    if MATRIX_SIZE == 6:
        NUM_OF_GRIDS = 400
    if MATRIX_SIZE == 7:
        NUM_OF_GRIDS = 1000


# initialize all the girds back to the beginning but, keeps the elites grids
# revolution - 100% start-over (happens once every 2000 generations)
def initialize(first_time, elites_list, revolution):
    for n in range(NUM_OF_GRIDS):
        temp_grid = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                if (i, j) in COORDINATES:
                    temp_grid[i, j] = COORDINATES[(i, j)]
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                if (i, j) not in COORDINATES:
                    num = random.randint(1, MATRIX_SIZE)
                    while num in temp_grid[i]:
                        num = random.randint(1, MATRIX_SIZE)
                    temp_grid[i, j] = num
        GRID_LIST.append(temp_grid)

    if not first_time and not revolution:
        for i in range(0, int(0.05 * NUM_OF_GRIDS)):
            if random.randint(0, 5) == 0:
                mutation(elites_list[i])
            GRID_LIST[i] = elites_list[i]


# Does crossover for two given grids.
# It picks a random number between 1 and rows size and taking first part of first grid
# and second part of second grid and concatenate them.
def cross_over(grid1, grid2):
    top_slice = random.randint(0, MATRIX_SIZE - 1)
    grid1 = grid1[:top_slice, :]
    grid2 = grid2[top_slice:, :]
    new_grid = np.concatenate((grid1, grid2), axis=0)
    return new_grid


# calculates the fitness of the given grid.
# it counts the number of success's -
# for every cell if it doesn't contradict it's row or colon, and for every correct sign
def fitness_func(grid_ft):
    fitness_score = 0
    for i in range(MATRIX_SIZE):
        row = []
        col = []
        for j in range(MATRIX_SIZE):
            row.append(grid_ft[i, j])
            col.append(grid_ft[j, i])
        diff_row = len(row) - len(set(row))
        diff_col = len(col) - len(set(col))
        fitness_score += diff_col + diff_row

    for sign in SIGN_COORDINATES:  # counts the correct signs
        first_index = sign[0]
        second_index = sign[1]
        if grid_ft[first_index[0] - 1, first_index[1] - 1] < grid_ft[second_index[0] - 1, second_index[1] - 1]:
            fitness_score += 1

    fitness_score = MATRIX_SIZE * MATRIX_SIZE + NUMBER_OF_GRATER_THAN_SIGNS - fitness_score
    return fitness_score


# Does a random mutation for a given grid.
# It does so by picking 2 randon indexes in the same row and swapping their values.
def mutation(grid_mu):
    row = random.randint(0, MATRIX_SIZE - 1)
    col1 = random.randint(0, MATRIX_SIZE - 1)
    col2 = random.randint(0, MATRIX_SIZE - 1)
    while (row, col1) in COORDINATES or (row, col2) in COORDINATES:
        row = random.randint(0, MATRIX_SIZE - 1)
        col1 = random.randint(0, MATRIX_SIZE - 1)
        col2 = random.randint(0, MATRIX_SIZE - 1)
    grid_mu[row, col1], grid_mu[row, col2] = grid_mu[row, col2], grid_mu[row, col1]


# return the second index of a number that repeat itself in the given array.
# return -1 if there isn't any duplicates.
def find_index_of_duplicate(arr):
    new_arr = []
    for i in range(0, len(arr)):
        if arr[i] not in new_arr:
            new_arr.append(arr[i])
        else:
            return i
    return -1


# return true if the sign are indeed true with the numbers it's related to.
def is_sign_correct(sign, grid):
    return True if grid[sign[0][0], sign[0][1]] > grid[sign[1][0], sign[1][1]] else False


# return true if the sign is vertical and false if it isn't.
def is_vertical(sign):
    return True if sign[0][0] != sign[1][0] else False


# Does the optimization for all the grids.
# going through every colon and if there is duplicates of numbers in that colon,
# it swaps the second duplicate in the row with the number that's missing in the colon from the same row.
def optimization():
    cell_set = {5}  # creating a set which we can use later without initializing it.
    cell_set.remove(5)
    for grid in GRID_LIST:
        num_of_switches = 0
        for i in range(MATRIX_SIZE):  # iterate through every colon.
            list_col = grid[:, i]
            index_of_duplicate = find_index_of_duplicate(list_col)
            if index_of_duplicate != -1:  # case there is a duplicate in the current colon - so we can optimize it.
                for m in range(1, MATRIX_SIZE + 1):
                    if m not in list_col:
                        row = grid[index_of_duplicate, :]
                        for k in range(0, len(row)):
                            if row[k] == m:
                                index_not_contain = k
                        grid[index_of_duplicate, index_not_contain], grid[index_of_duplicate, i] \
                            = grid[index_of_duplicate, i], grid[index_of_duplicate, index_not_contain]
                        num_of_switches += 1
                        break
            # checking if after the optimization the grid reached completeness
            if fitness_func(grid) == MATRIX_SIZE * MATRIX_SIZE + NUMBER_OF_GRATER_THAN_SIGNS:
                return True, grid


# Return true if the grid is already in the grids list.
def grid_already_in(grids, new_grid):
    for grid in grids:
        if np.array_equal(grid, new_grid):
            return True
    return False


# Counts the number of equal grids
# Using set because set doesn't have duplicates, so adding the same grid to the set, it remains the same size.
def equals_grids():
    n = 0
    duplicate_set = {5}  # creating a set which we can use later without initializing it.
    duplicate_set.discard(5)
    for i in range(len(GRID_LIST)):
        for j in range(len(GRID_LIST)):
            if i != j:
                if np.array_equal(GRID_LIST[i], GRID_LIST[j]):
                    n += 1
                    duplicate_set.add(i)
    return len(duplicate_set)


# returns an array that when you random a value from the array,
# it'll return an index of a random grid while taking into consideration his favoritism.
def get_favoritism():
    fitness_scores = []
    for i in range(0, len(GRID_LIST)):
        fitness_scores.append(fitness_func(GRID_LIST[i]))
    sum_fitness = int(np.sum(fitness_scores))

    # sums all the grids fitness and then storing for each grid his favoritism in an array
    favoritism = np.zeros(sum_fitness, dtype=int)
    for i in range(0, len(GRID_LIST)):
        fitness_scores[i] = (fitness_scores[i] / sum_fitness)

    d = 0
    for i in range(0, len(fitness_scores)):
        for j in range(0, int(fitness_scores[i] * sum_fitness)):
            favoritism[d] = i
            d += 1
    return favoritism, len(favoritism)


# returns 5% of the grids which has the best fitness
def get_elites():
    elitism_list = []
    fitness_scores = []
    for i in range(0, len(GRID_LIST)):
        fitness_scores.append(fitness_func(GRID_LIST[i]))
    i = 0.05 * NUM_OF_GRIDS
    while i > 0:
        max_val = max(fitness_scores)
        index_max = fitness_scores.index(max_val)
        curr_grid = GRID_LIST[index_max]
        if not grid_already_in(elitism_list, curr_grid):
            elitism_list.append(curr_grid)
            i -= 1
        fitness_scores[index_max] = 0

    return elitism_list


# for all the grids, for one generation, this function manage the optimization, mutations, cross_overs and elites.
def generation():
    favoritism, fav_size = get_favoritism()
    new_grids = []
    global GRID_LIST
    # goes over all the grids and does crossovers and mutations
    for i in range(0, int(0.95 * NUM_OF_GRIDS)):
        # Picking the right index of grids from favoritism array
        first_grid_index = favoritism[random.randint(0, fav_size - 1)]
        second_grid_index = favoritism[random.randint(0, fav_size - 1)]
        new_grid = cross_over(GRID_LIST[first_grid_index], GRID_LIST[second_grid_index])
        if random.randint(0, 5) == 0:
            mutation(new_grid)
        new_grids.append(new_grid)
        if fitness_func(new_grid) == MATRIX_SIZE * MATRIX_SIZE + NUMBER_OF_GRATER_THAN_SIGNS:
            return True, new_grid

    for grid in get_elites():  # saves 5% elites grids for next generation.
        if fitness_func(grid) == MATRIX_SIZE * MATRIX_SIZE + NUMBER_OF_GRATER_THAN_SIGNS:
            return True, grid
        new_grids.append(grid)

    for i in range(0, len(new_grids)):
        GRID_LIST[i] = new_grids[i]

    if METHOD == "Lamarck" or METHOD == "Darvin":  # if Darvin or Lamarck methods, then do optimizations.
        temp_grids = GRID_LIST
        optimization()
        if METHOD == "Dravin":
            GRID_LIST = temp_grids
    return False, None


# if it's vertical sign - returns true if top is greater than bottom
# if it's horizontal sign - returns true if left is greater than right
def is_left_or_is_top_bigger(sign):
    if is_vertical(sign):
        if sign[0][0] < sign[1][0]:
            return True
        return False
    else:
        if sign[0][1] < sign[1][1]:
            return True
        return False


# Prints a simple gui of the completeness grid with the signs next to the related numbers.
def show_gui(grid):
    gui = np.zeros((2 * MATRIX_SIZE, 2 * MATRIX_SIZE), dtype='U1')
    for i in range(0, 2 * MATRIX_SIZE):  # add all the numbers to the gui array
        for j in range(0, 2 * MATRIX_SIZE):
            if i % 2 == 1 or j % 2 == 1:  # spot of sign
                gui[i, j] = ' '
            else:  # spot of number
                gui[i, j] = str(grid[int(i / 2), int(j / 2)])
    for sign in SIGN_COORDINATES:  # add all the signs to the array
        if is_vertical(sign):
            if is_left_or_is_top_bigger(sign):
                gui[sign[0][0] + sign[1][0] - 2, sign[0][1] + sign[1][1] - 2] = 'v'
            else:
                gui[sign[0][0] + sign[1][0] - 2, sign[0][1] + sign[1][1] - 2] = '^'
        else:
            if is_left_or_is_top_bigger(sign):
                gui[sign[0][0] + sign[1][0] - 2, sign[0][1] + sign[1][1] - 2] = '>'
            else:
                gui[sign[0][0] + sign[1][0] - 2, sign[0][1] + sign[1][1] - 2] = '<'

    for i in range(0, 2 * MATRIX_SIZE - 1):  # prints the gui array with dividers between.
        print('|', end='')
        for j in range(0, 2 * MATRIX_SIZE - 1):
            print(gui[i, j], end='|')
        print()


def run():
    run_flag = True
    num_of_generations = 0
    revolution_counter = 2000  # every number of revolution_counter generations, 100% start-over
    # loop that runs until we get a completeness grid.
    while run_flag:
        is_completeness, final_grid = generation()
        if is_completeness:  # case a completeness grid was found.
            fitness_func(final_grid)
            show_gui(final_grid)
            run_flag = False
        else:
            num_of_generations += 1
            # case many grids are alike, meaning rapid convergence happened, then we start over.
            if num_of_generations % 200 == 199:
                num_of_equal_grids = equals_grids()

                if num_of_generations > revolution_counter:
                    initialize(False, get_elites(), True)
                    revolution_counter += 2000
                else:
                    if num_of_equal_grids >= 40:  # number of duplicates became too high (might be local maximum)
                        initialize(False, get_elites(), False)

    return num_of_generations


def main():
    start_up()  # initialize all inputs in better structs.
    initialize(True, None, False)  # initialize all the girds
    num_g = run()  # start the loop which will return the completeness grid.
    print("number of generations: " + str(num_g))


if __name__ == "__main__":
    main()
