from scipy.spatial import ConvexHull
import numpy as np
try:
    import cupy as cp
except ImportError:
    print("cupy not installed, using numpy instead")
    import numpy as cp
from sklearn.metrics import pairwise_distances


def approximate_convex_hull(points, k):
    hull = [points[0]]
    while len(hull) < k and len(points) > 0:
        centroid = np.mean(hull, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        farthest = np.argmax(distances)
        hull.append(points[farthest])
        points = np.delete(points, farthest, axis=0)

    # return the hull as an array
    return np.array(hull)


def calc_avg_dist(distance):
    return cp.asnumpy(cp.sum(cp.asarray(distance), axis=1) / len(distance))


def distance_matrix_no_loops(matrix):
    distance_matrix = cp.zeros((matrix.shape[0], matrix.shape[0]))
    # Use broadcasting to calculate the Euclidean distance between all pairs of points.
    distance_matrix = cp.linalg.norm(matrix[:, np.newaxis] - matrix, axis=-1)

    return cp.asnumpy(distance_matrix), calc_avg_dist(distance_matrix)


def calc_avg_distnp(distance):
    return np.sum(np.array(distance), axis=1) / len(distance)


def distance_matrix_no_loopsnp(matrix):
    # distance_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    # Use broadcasting to calculate the Euclidean distance between all pairs of points.
    # distance_matrix = np.linalg.norm(matrix[:, np.newaxis] - matrix, axis=-1)
    distance = pairwise_distances(matrix, matrix, n_jobs=-1)
    return distance, calc_avg_dist(distance)


# cpu
def add_cluster(
    data,
    initial_point=(True, 0, 0),
    avg_dist_recurrsion="LOL",
    ratio=None,
    hull_all_area=None,
):
    # get furthest point
    distance_matrix, avg_dist = distance_matrix_no_loopsnp(np.array(data))
    if initial_point[0]:
        distance_max = distance_matrix[:, np.argmax(avg_dist)]
        distance_min = distance_matrix[:, np.argmin(avg_dist)]
    else:
        avg_dist = avg_dist_recurrsion
        distance_min = distance_matrix[:, np.argmin(avg_dist)]
        distance_max = initial_point[1]
    new_cluster = [1 if x < 0 else 0 for x in distance_max - distance_min]
    #     if sum(new_cluster) < 5 or sum(np.logical_not(new_cluster)) < 5:
    #         avg_dist[np.argmax(avg_dist)]=np.min(avg_dist)
    #         data_copy = data.copy()
    #             #data_copy[np.argmax(avg_dist)] = np.array([0,0])
    #         new_cluster = add_cluster(data_copy, initial_point= (False,distance_matrix[:,np.argmax(avg_dist)]), avg_dist_recurrsion=avg_dist, ratio=None,hull_all_area=hull_all_area)[0]
    distance_max = np.delete(distance_max, np.argmax(avg_dist))
    distance_min = np.delete(distance_min, np.argmin(avg_dist))
    all_log = []

    hull = ConvexHull(data[new_cluster], qhull_options="QJ")
    if ratio is None or True:
        try:
            hull_all = ConvexHull(data[np.logical_not(new_cluster)], qhull_options="QJ")
        except:
            print(len(data), sum(new_cluster), sum(np.logical_not(new_cluster)), "HI")
            return (np.array(new_cluster), distance_min, distance_max, hull.area, 0)

        hull_all_area = hull_all.area
        ratio = np.log(len(data) - sum(new_cluster)) / hull_all.area
        # ratio = np.log(len(data)-sum(new_cluster))/np.max(distance_min)

    new_center = None

    sim_new_indecies = None
    count_log = 0

    while (
        np.log(sum(new_cluster)) / hull.area <= ratio
    ):  # and not math.isclose(hull.area, hull_all.area, rel_tol=1e-5):
        all_log.append(np.log(sum(new_cluster)) / hull.area)
        sim_new_indecies = np.where(np.array(new_cluster) == 0)[0]
        new_sim = distance_matrix[:, sim_new_indecies]
        new_sim2 = new_sim[sim_new_indecies, :]
        new_center = np.argmin(calc_avg_distnp(np.array(new_sim2)))
        print(sim_new_indecies[new_center])
        distance_max = distance_matrix[:, sim_new_indecies[new_center]]
        distance_min = distance_matrix[:, np.argmin(avg_dist)]
        new_cluster = [1 if x < 0 else 0 for x in distance_max - distance_min]

        # sometimes when oscilating, it doesn't stop at the max ratio, change that to get the best cluster
        # ###################### to do ###############

        if len(all_log) > 5:
            #             if all_log[-1] == all_log[-3]:
            if all_log[-1] == all_log[-3]:
                break
        if sum(new_cluster) == 0:
            print("I GOT PROCED")
            avg_dist[np.argmax(avg_dist)] = np.min(avg_dist)
            data_copy = data.copy()
            # data_copy[np.argmax(avg_dist)] = np.array([0,0])

            new_cluster = add_cluster(
                data_copy,
                initial_point=(False, distance_matrix[:, np.argmax(avg_dist)]),
                avg_dist_recurrsion=avg_dist,
                ratio=None,
                hull_all_area=hull_all_area,
            )[0]
        hull = ConvexHull(data[new_cluster], qhull_options="QJ")

    return (np.array(new_cluster), distance_min, distance_max, hull.area, hull_all_area)


def the_rich_gets_richer(data, number_of_clusters, divisable_number=100):
    clusters = np.zeros((len(data),))
    cluster_size = {}
    cluster_center = {}
    cluster_main, distance_min, distance_max, area0, all_area = add_cluster(data)

    cluster_size[0] = np.max(distance_min)
    cluster_size[1] = np.max(distance_max)

    cluster_center[0] = data[np.argmin(distance_min)]
    cluster_center[1] = data[np.argmin(distance_max)]
    #     cluster_size[0] = np.max(distance_min)
    #     cluster_size[1] = np.max(distance_max)
    biggest_cluster = max(cluster_size, key=lambda x: cluster_size[x])
    cs_c = cluster_size.copy()
    while len(data[np.where(np.array(cluster_main) == biggest_cluster)[0]]) < 100:
        cs_c[biggest_cluster] = 0.000000000000001
        biggest_cluster = max(cs_c, key=lambda x: cs_c[x])
    data_temp = data[np.where(np.array(cluster_main) == biggest_cluster)[0]]
    ratio = np.log(len(data)) / np.max(distance_min)
    for i in range(number_of_clusters - 2):
        cluster, distance_min, distance_max, area0, all_area = add_cluster(
            data_temp, ratio=ratio, hull_all_area=all_area
        )
        cluster = cluster * (i + 2)
        cluster[np.where(cluster == 0)[0]] = biggest_cluster
        cluster_size[biggest_cluster] = np.max(distance_min)
        cluster_size[i + 2] = np.max(distance_max)
        cluster_center[biggest_cluster] = data[np.argmin(distance_min)]
        cluster_center[i + 2] = data[np.argmin(distance_max)]
        indicies = np.where(np.array(cluster_main) == biggest_cluster)[0]
        cluster_main[indicies] = cluster
        cluster_ind = np.where(np.array(cluster) == 1)[0]
        biggest_cluster = max(cluster_size, key=lambda x: cluster_size[x])
        cs_c = cluster_size.copy()
        while (
            len(data[np.where(np.array(cluster_main) == biggest_cluster)[0]])
            < divisable_number
        ):
            cs_c[biggest_cluster] = 0.000000000000001
            biggest_cluster = max(cs_c, key=lambda x: cs_c[x])
        data_temp = data[np.where(np.array(cluster_main) == biggest_cluster)[0]]
        cluster_prev = cluster.copy()

    return cluster_main, cluster_size, cluster_center
