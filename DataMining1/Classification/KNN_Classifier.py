import numpy as np
from heapq import heappush, heappop


""" KNN Implementation """

def EuclideanDistance(v1, v2, dims):
    distance = 0.0
    for i in range(0, dims):
        distance += np.square(v1[i] - v2[i])
    return np.sqrt(distance)


class KNN:
	def __init__(self):
		self.points = []  # list of 2-tuples all points learned and their categories
		self.categories = set()
	
	def fit(self, train_set, train_categories):
		self.categories = set(train_categories)
		for i in range(0, len(train_set)):
			self.points.append((train_set[i], train_categories[i]))

	def predict(self, test_set, K):
		prediction = []
		for newpoint in test_set:
			prediction.append(self.predict_for_one(newpoint, K))
		return prediction
	
	def predict_for_one(self, new_point, K):
		dim = len(new_point)
		if len(self.points[0][0]) != dim:
			print 'Error at predict_for_one: new point has wrong dimensions'
			return None
		heap = []               # minheap
		for point in self.points:
			heappush(heap, (EuclideanDistance(point[0], new_point, dim), point[1]))
		category_count = {c:0 for c in self.categories}
		for i in range(0, K):   # only pop top-K (smallest K distances)
			_, c = heappop(heap)
			category_count[c] += 1
		max = -1
		maxcat = None
		for cat in self.categories:
			if category_count[cat] > max:
				max = category_count[cat]
				maxcat = cat
		return maxcat
