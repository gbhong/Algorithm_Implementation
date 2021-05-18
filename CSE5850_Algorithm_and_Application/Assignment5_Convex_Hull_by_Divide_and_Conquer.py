# Assignment 5_Algorithm and App.
# 2020312086 Hong Gibong

# Q1 Convex Hull with Divide-and-Conquer and Brute-Force
import random
import time
import matplotlib.pyplot as plt

def construct_points(N:int):
    points = [(random.random(), random.random()) for _ in range(N)]
    return sorted(set(points), key=lambda x:x[0])

def cross(a:tuple, b:tuple, c:tuple):
    '''
        Computes the sign perpendicular distance of a 2d point c from a vector ab. The sign indicates the direction of c relative to ab.
        A Positive value means point c is above ab (to the left), while a negative value means that point c is below ab (to the right).
        0 means that all three points are on the same straight line.
    '''
    return (a[0]*b[1] + b[0]*c[1] + c[0]*a[1]) - (a[1]*b[0] + b[1]*c[0] + c[1]*a[0])

def convex_hull_dc(points:list):
    '''
        Constructs the convex hull of a set of 2D points using a divide-and-conquer strategy.
        The algorithm explits the geometric properties of the problem by repeatedly
        partitioning the set of points into smaller hulls, and finding the convex hull of
        these smaller hulls.
        The union of the convex hull from smaller hulls is the solution to the convex hull of the larger problem.
    '''
    N = len(points)
    left_most, right_most = points[0], points[-1]

    convex_set = {left_most, right_most}
    upper_hull, lower_hull = [], []

    for i in range(1, N-1):
        det = cross(left_most, right_most, points[i])

        if det > 0:
            upper_hull.append(points[i])
        elif det < 0:
            lower_hull.append(points[i])

    construct_hull(upper_hull, left_most, right_most, convex_set)
    construct_hull(lower_hull, right_most, left_most, convex_set)

    return sorted(convex_set)

def construct_hull(points:list, left:tuple, right:tuple, convex_set:set):
    '''
        Returns nothing, only updates the state of convex-set
    '''
    if points:
        extreme_point = None
        extreme_point_dist = float('-inf')
        candidate_points = []

        for p in points:
            det = cross(left, right, p)

            if det > 0:
                candidate_points.append(p)

                if det > extreme_point_dist:
                    extreme_point_dist = det
                    extreme_point = p

            if extreme_point:
                construct_hull(candidate_points, left, extreme_point, convex_set)
                convex_set.add(extreme_point)
                construct_hull(candidate_points, extreme_point, right, convex_set)

def convex_hull_bf(points:list):
    '''
        Computes the convex hull of a set of 2D points by brute-force algorith
    '''
    N = len(points)
    upper_hull, lower_hull = [], []

    # record upper hull bound using maximum value between the left-most, right-most points
    upper_hull_bound, lower_hull_bound = max(points[0][1], points[-1][1]), min(points[0][1], points[-1][1])

    for i in range(N-1):
        for j in range(i+1, N):
            points_left_of_ij = points_right_of_ij = False
            ij_part_of_convex_hull = True
            for k in range(N): # for all other points except point[i] and point[j]
                if k != i and k != j:
                    res_k = cross(points[i], points[j], points[k])
                    if res_k > 0:
                        points_left_of_ij = True
                    elif res_k < 0:
                        points_right_of_ij = True
                    else:
                        # for this case, point k is on the same line with point i and point j
                        # if point k is to the left of point i, or it is to the right of point j,
                        # then point i and point j cannot be part of the convex hull of given points.
                        if points[k][0] < points[i][0] or points[k][0] > points[j][0]:
                            ij_part_of_convex_hull = False
                            break

                # this means that given vector ij cannot be part of convex hull
                if points_left_of_ij and points_right_of_ij:
                    ij_part_of_convex_hull = False
                    break

            # determine whether given vector ij belongs to upper hull or lower hull
            if ij_part_of_convex_hull:
                if min(points[i][1], points[j][1]) >= upper_hull_bound:
                    upper_hull.append([points[i], points[j]])
                else:
                    lower_hull.append([points[i], points[j]])

    return (sorted(upper_hull), sorted(lower_hull))

time_records = []
input_list = [1e3, 3e3, 5e3, 7e3, 9e3]
for n in input_list:
    points = construct_points(int(n)) # initialize 2D coordinates

    t_start = time.time()
    convex_points = convex_hull_dc(points)
    elapsed_time = time.time() - t_start

    print(n, '\t', elapsed_time)
    time_records.append(elapsed_time)

n = [time_records[0]*(i/1000) for i in input_list]
np = [time_records[0]*(i/1000)*0.5 for i in input_list]

plt.figure(figsize=(12,6))
plt.plot(range(len(input_list)), time_records, 'b', label='Elapsed Time')
plt.plot(range(len(input_list)), n, 'r', label='O(n)')
plt.plot(range(len(input_list)), np, 'r', label='O(0.5*n)')

plt.title('Comparison on Elapsed Time following input size')
plt.xlabel('Input Size')
plt.ylabel('Elapsed Time')
plt.xticks(ticks=range(len(input_list)), labels=input_list)
plt.legend()
plt.show()
plt.savefig('./assignment5_q1_plot.png', dpi=300)

# visualize the resulting convex hull, only in the case when n = 9000
cx, cy = [], []
for x, y in convex_points:
    cx.append(x)
    cy.append(y)
plt.scatter(cx, cy)

ax, ay = [], []
for x, y in set(points)-set(convex_points):
    ax.append(x)
    ay.append(y)

plt.figure(figsize=(12,8))
plt.scatter(cx, cy, c='b')
plt.scatter(ax, ay, c='r')
