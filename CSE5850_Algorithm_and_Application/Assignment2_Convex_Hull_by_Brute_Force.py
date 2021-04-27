# Assignment 2_Algorithm and App.
# 2020312086 Hong Gibong

# 2. Convex-Hull Problem

import random
import time
import matplotlib.pyplot as plt

def cross(a:tuple, b:tuple, c:tuple):
    '''
    Computes the sign perpendicular distance of a 2d point c from a vector ab. The sign indicates the direction of c relative to ab.
    A Positive value means point c is above ab (to the left), while a negative value means that point c is below ab (to the right).
    0 means that all three points are on the same straight line.
    '''
    return (a[0]*b[1] + b[0]*c[1] + c[0]*a[1]) - (a[1]*b[0] + b[1]*c[0] + c[1]*a[0])

def convex_hull_bf(N:int):
    '''
    Computes the convex hull of a set of 2D points.

    ARGS:
         - N: number of (x,y) pairs
    '''
    # Generate multiple random (x,y) coordinates excluding duplicates.
    points = [(random.random(), random.random()) for _ in range(N)]

    # Check start time
    t_start = time.time()

    # Sort the points and remove duplicates
    points = sorted(set(points), key=lambda x:x[0])
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

    return (sorted(upper_hull), sorted(lower_hull)), time.time() - t_start

t_list = []
input_list = [1000,3000,5000,7000,9000]
for n in input_list:
    convex_lst, elapsed_t = convex_hull_bf(int(n))
    print(n, '\t', elapsed_t)
    t_list.append(elapsed_t)

n3 = [t_list[0]*(i/1000)**3 for i in input_list]
n3p = [t_list[0]*(i/1000)**3*0.5 for i in input_list]

plt.figure(figsize=(12,6))
plt.plot(range(len(input_list)), t_list, 'b', label='Elapsed Time')
plt.plot(range(len(input_list)), n3, 'r', label='O(n^3)')
plt.plot(range(len(input_list)), n3p, 'r', label='O(0.5*n^3)')

plt.title('Comparison on Elapsed Time following input size')
plt.xlabel('Input Size')
plt.ylabel('Elapsed Time')
plt.xticks(ticks=range(len(input_list)), labels=input_list)
plt.legend()
# plt.show()
plt.savefig('./assignment2_plot.png', dpi=300)
