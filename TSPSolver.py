#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from heapq import heappush, heappop
import copy



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		# This is the greedy algorithm that helps find the initial BSSF.
		# Sets up the unvisited cities and creates the results set.
		# Sets counters and indices to 0 and sets the initial BSSF to none.
		results = {}
		cities = self._scenario.getCities()
		unvisited = []
		for c in cities:
			unvisited.append(c._index)
		num_cities = len(cities)
		foundTour = False
		start_index = 0
		city_index = 0
		count = 0
		bssf = None
		# Places the first city in the route array and removes it from the unvisited.
		# Sets the start time.
		route = [cities[0]]
		unvisited.remove(cities[0]._index)
		starttime = time.time()
		# While the start index is less than the length of the cities array
		# And the time is within allowance
		while start_index < len(cities) and time.time() - starttime < time_allowance:
			# Set the current city to the city at that point in the array
			current_city = cities[city_index]
			best_city = None
			# For each city in the unvisited array
			for city in unvisited:
				# If there is no best city, or the current city's cost to the city we're looking at in the array
				# is greater than the cost to the best city from the current one, then
				# set that as the new best city.
				if best_city == None or current_city.costTo(cities[city]) < current_city.costTo(best_city):
					best_city = cities[city]
			# Remove the best city from unvisited and add it to the route.
			# Update the index.
			unvisited.remove(best_city._index)
			route.append(best_city)
			city_index = best_city._index
			# If the unvisited array is at 0 length
			# Set the solution
			if len(unvisited) == 0:
				sol = TSPSolution(route)
				# If the cost is less than infinity, then we've found a tour
				# Add that to the number of solutions
				# If there isn't a BSSF, set the BSSF
				# If there is and the current solution cost is less than the BSSF cost
				# set the solution as the new BSSF
				if sol.cost < np.inf:
					foundTour = True
					count += 1
					if bssf == None:
						bssf = sol
					elif sol.cost < bssf.cost:
						bssf = sol
				# Increment the index.
				start_index += 1
				# If the index is less than the length of the cities array
				# then for each city in the city array
				# add that city to unvisited.
				# Set the route to that starting city
				# Set the city index to that start index and remove it from unvisited
				if start_index < len(cities):
					for city in cities:
						unvisited.append(city._index)
					route = [cities[start_index]]
					city_index = start_index
					unvisited.remove(cities[start_index]._index)

		# Set up the results set.
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - starttime
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
	

	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		# Sets up all the initial values, the cities array, the number of cities,
		# starting indices, counters and cities.
		# and estimates the BSSF using the greedy algorithm.
		results = {}
		cities = self._scenario.getCities()
		n_cities = len(cities)
		foundTour = False
		bssf = self.estimateBSSF(cities)
		pruned = 0
		# Get initial distances
		city_dist = self.calcCityDist(cities)
		priorityQueue = []
		total = 0
		count = 0
		start_time = time.time()
		starting_city_index = 0
		start_lower_bound = 0
		# Set up the TSP problem (the start state) using a starting lower bound and a copy of the city distances, as well as the
		# starting index. Calculate the lower bound and push the problem into a heap.
		# increments the number of total states.
		starting_prob = TSPProb(start_lower_bound, copy.deepcopy(city_dist), [starting_city_index])
		starting_prob.calcLB()
		heappush(priorityQueue, starting_prob)
		maxheap = len(priorityQueue)
		total = total + 1
		# While the length of the queue is greater than 0 and the time is less than the allowance
		# Set the current problem
		while len(priorityQueue) > 0 and time.time() - start_time < time_allowance:
			prob = heappop(priorityQueue)
			# If the LB of the problem is greater than the cost of the bssf, don't consider it
			# Add to the pruned count
			if prob.LB >= bssf.cost:
				pruned = pruned + 1
			else:
			# A complete solution has been found here
			# if the length of the tour is equal to the number of cities
			# create a route array and append the cities in the tour to it
			# Set the solution and check with the BSSF cost
			# if it's less, then set the solution as the new BSSF and add to the solution count
				if len(prob.tour) == n_cities:
					route = []
					for i in range(len(prob.tour)):
						route.append(cities[prob.tour[i]])
					sol = TSPSolution(route)
					if sol.cost < bssf.cost:
						bssf = sol
						count = count + 1
			# If the length of the tour is not the same as the number of cities
				else:
					for i in range(n_cities):
						# For each of the number of the cities
						# if it's not in the tour (if it's unvisited), create a new tour with a copy of the problem tour
						# and increment the total states
						# Create a sub problem and expand it
						if i not in prob.tour:
							new_tour = copy.deepcopy(prob.tour)
							total = total + 1
							sub_prob = TSPProb(prob.LB, copy.deepcopy(prob.distances), new_tour)
							sub_prob.travel(sub_prob.tour[len(prob.tour) - 1], i)
							sub_prob.calcLB()
							# Check the subproblem's LB against the bssf cost and update if necessary
							if sub_prob.LB < bssf.cost:
								heappush(priorityQueue, sub_prob)
								if len(priorityQueue) > maxheap:
									maxheap = len(priorityQueue)
							else:
								pruned = pruned + 1
		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = maxheap
		results['total'] = total
		results['pruned'] = pruned
		return results

	# Calculates the city distances
	def calcCityDist(self, cities):
		distances = np.zeros( (len(cities), len(cities)), dtype=float)
		for i in range(len(cities)):
			for j in range(len(cities)):
				distances[i][j] = cities[i].costTo(cities[j])
		return distances

	# Grabs an estimated BSSF from the greedy algorithm
	def estimateBSSF(self, cities):
		bssf = self.greedy()['soln']
		return bssf


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):

		pass

# This is a class that creates an object for the TSP problem as a whole.
# It contains class objects for the distances array, the lower bound and the tour.
# It contains functions to reduce the matrix by rows and columns and checks for infinities.
# It also calculates the lower bound.
class TSPProb:
	def __init__(self, start_LB, unreduced_dist, tour):
		self.distances = unreduced_dist
		self.LB = start_LB
		self.tour = tour

# Calculates the lower bound. Checks if the distance matrix is reduced and adds lowest value in each row/column to the total
# lower bound.
	def calcLB(self):
		for row in range(len(self.distances)):
			if not self.rowHasZero(self.distances, row) and not self.isRowInfinite(self.distances, row):
				to_be_added = self.reduceRow(self.distances, row)
				self.LB = self.LB + to_be_added
		for col in range(len(self.distances)):
			if not self.colHasZero(self.distances, col) and not self.isColInfinite(self.distances, col):
				to_be_added = self.reduceCol(self.distances, col)
				self.LB = self.LB + to_be_added

# Checks if a row has a zero
	def rowHasZero(self, matrix, row):
		for col in range(len(matrix[0])):
			if matrix[row][col] == 0:
				return True
		return False
#Checks if a row has only infinite values
	def isRowInfinite(self, matrix, row):
		for col in range(len(matrix[0])):
			if matrix[row][col] != float('inf'):
				return False
		return True
# Checks if a column has a zero
	def colHasZero(self, matrix, col):
		for row in range(len(matrix[0])):
			if matrix[row][col] == 0:
				return True
		return False
# Checks if a column has only infinite values
	def isColInfinite(self, matrix, col):
		for row in range(len(matrix[0])):
			if matrix[row][col] != float('inf'):
				return False
		return True
# Reduces a row by finding the least value and subtracting that from
# each value in the row.
	def reduceRow(self, matrix, row):
		least = float('inf')

		# Find the least number in the row
		for col in range(len(matrix[0])):
			if matrix[row][col] < least:
				least = matrix[row][col]
		# Reduce the row
		for col in range(len(matrix[0])):
			matrix[row][col] = matrix[row][col] - least

		return least

# Reduces a column by finding the least value and subtracting that from
# each value in the column
	def reduceCol(self, matrix, col):
		least = float('inf')

		# Find the least number in the column
		for row in range(len(matrix[0])):
			if matrix[row][col] < least:
				least = matrix[row][col]
		# Reduce the row
		for row in range(len(matrix[0])):
			matrix[row][col] = matrix[row][col] - least

		return least

# Adds the destination city to the tour and appends the distance from
# the cities to the LB. Sets the corresponding row at the city from, and the column at the city to all to infinity.
	def travel(self, city_from, city_to):
		self.tour.append(city_to)
		self.LB = self.LB + self.distances[city_from][city_to]
		self.setMatrixRow(self.distances, city_from, float('inf'))
		self.setMatrixCol(self.distances, city_to, float('inf'))
		self.distances[city_to][city_from] = float('inf')

# Sets a value to a whole row
	def setMatrixRow(self, matrix, row, value):
		matrix[row] = [value] * len(matrix[row])
# Sets a value to a whole column
	def setMatrixCol(self, matrix, col, value):
		for row in range(len(matrix)):
			matrix[row][col] = value

# Prioritizes the tour length to find a solution faster, and then
# looks at the lower bound.
	def __lt__(self, other):
		if len(self.tour) == len(other.tour):
			return self.LB < other.LB
		return len(self.tour) > len(other.tour)



