#!/usr/bin/env python3
'''Creates a det map from beam map pixel locations for a detector array.'''

from copy import copy
import numpy as np
import matplotlib.pyplot as plt


def neighbor_debug_plot(pixels, point, dist, nearest_neighbors):
    '''Show a plot with the nearest neighbor pixels highlighted.'''
    plt.scatter(pixels[:, 0], pixels[:, 1], s=3, alpha=.5)
    plt.scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1],
                color="teal", label='nearest neighbors')
    plt.scatter(point[0], point[1], color="blue", label='start pixel')
    circ = plt.Circle(point, dist, color='g', fill=False)
    plt.xlim(point[0] - 1, point[0] + 1)
    plt.ylim(point[1] - 1, point[1] + 1)
    axis = plt.gca()
    axis.add_patch(circ)
    plt.legend()


def get_nearest_neighbors(pixels, point, dist, debug=0, return_dists=False):
    '''Find all pixels in /pixels/ within /dist/ of /point/.'''
    pixels = np.array(pixels)
    all_dists = dists_from_point(pixels, point)
    close_dists = (all_dists < dist)
    nearest_neighbors = pixels[close_dists]
    if debug > 0:
        neighbor_debug_plot(pixels, point, dist, nearest_neighbors)
        plt.show()
    if return_dists:
        return nearest_neighbors, all_dists[close_dists]
    return nearest_neighbors


def dists_from_point(pixels, point):
    '''Return the distance of each pixel in /pixels/ to /point/.'''
    pixels = np.array(pixels)
    x_dists = pixels[:, 0] - point[0]
    y_dists = pixels[:, 1] - point[1]
    return np.sqrt(x_dists ** 2 + y_dists ** 2)


def calc_centroid(pixels):
    '''Calculate the centroid of all points in /pixels/'''
    if len(pixels) == 0:
        pixels = [[np.nan, np.nan]]
    return np.mean(pixels, axis=0)


def pixel_debug_plot(close_pixels, centroid, max_centroid_dist,
                     circ_color='g'):
    '''Show a plot with the centroid and pixels within max_centroid_dist.'''
    close_pixels = np.array(close_pixels)
    plt.scatter(close_pixels[:, 0],
                close_pixels[:, 1], label='close pixels')
    plt.scatter(centroid[0], centroid[1],
                color="black", label='centroid')
    circ = plt.Circle(centroid, max_centroid_dist, color=circ_color,
                      fill=False)
    axis = plt.gca()
    axis.add_patch(circ)
    plt.legend()


def create_unified_pixel(close_pixels, max_centroid_dist, debug=0, verbose=2):
    '''Create a unified pixel from close together points in /close_pixels/.

    Returns the centroid of /close_pixels/ as long as each of them are less
    then /max_centroid_dist/ from the centroid.
    '''
    centroid = calc_centroid(close_pixels)
    dists_from_centroid = dists_from_point(close_pixels, centroid)

    if debug > 0:
        pixel_debug_plot(close_pixels, centroid, max_centroid_dist)
        plt.show()
    if np.max(dists_from_centroid) <= max_centroid_dist:
        return centroid, close_pixels, 0
    # if len(close_pixels) <= 4:
    #     print('<= 4 close pixels. Returning these as the pixel')
    #     print(80 * '#')
    #     return centroid, close_pixels, 1
    # Otherwise remove largest distance pixel and try again.
    if verbose > 1:
        print("pruning furthest neighbor to be within /max_centroid_dist/")
    good_subset = dists_from_centroid < np.max(dists_from_centroid)
    new_centroid = calc_centroid(close_pixels[good_subset])
    dists_from_new_centroid = dists_from_point(close_pixels[good_subset],
                                               new_centroid)
    if debug > 0:
        pixel_debug_plot(close_pixels[good_subset], new_centroid,
                         max_centroid_dist, circ_color='r')
        plt.show()
    if verbose > 1:
        print(80 * '#')
    if np.max(dists_from_new_centroid) <= max_centroid_dist:
        return new_centroid, close_pixels[good_subset], 2
    # Prune one more time and show 3 as a very uncertain value here
    if verbose > 1:
        print('pruning one more furthest neighbor')
    new_good_subset = dists_from_new_centroid < np.max(dists_from_new_centroid)
    new_centroid = calc_centroid(close_pixels[good_subset][new_good_subset])
    return new_centroid, close_pixels[good_subset][new_good_subset], 3


def get_pixel(pixel_list, point, neighbor_dist=.36, max_centroid_dist=.22,
              debug=0, verbose=0):
    '''Find the points in /pixel_list/ that compromise a pixel with /point/.'''
    nearest_neighbors = get_nearest_neighbors(pixel_list, point,
                                              dist=neighbor_dist, debug=debug)
    centroid, close_pixels, return_type = create_unified_pixel(
        nearest_neighbors, max_centroid_dist=max_centroid_dist, debug=debug,
        verbose=verbose)
    return centroid, close_pixels, return_type


def get_pixel_debug(pixel_list, point, neighbor_dist=.36,
                    max_centroid_dist=.22):
    '''Find the points in /pixel_list/ that compromise a pixel with /point/.'''
    pixel_list = np.array(pixel_list)
    nearest_neighbors = get_nearest_neighbors(pixel_list, point,
                                              dist=neighbor_dist, debug=0)
    centroid, close_pixels, return_type = create_unified_pixel(
        nearest_neighbors, max_centroid_dist=max_centroid_dist, debug=0)
    if centroid is None:
        print('returned None. Debugging old fashioned way.')
        centroid, close_pixels, return_type = create_unified_pixel(
            nearest_neighbors, max_centroid_dist=max_centroid_dist, debug=1)
        plt.subplots(figsize=(6, 3))
        neighbor_debug_plot(np.array(pixel_list), point, neighbor_dist,
                            nearest_neighbors)
        plt.show()
        return None, None, None

    plt.subplots(1, 2, figsize=(8, 3))
    plt.subplot(1, 2, 1)
    neighbor_debug_plot(np.array(pixel_list), point, neighbor_dist,
                        nearest_neighbors)
    plt.subplot(1, 2, 2)
    pixel_debug_plot(close_pixels, centroid, max_centroid_dist,
                     circ_color='red')
    plt.scatter(pixel_list[:, 0], pixel_list[:, 1], s=3, color='tab:orange')
    plt.xlim(centroid[0] - 1, centroid[0] + 1)
    plt.ylim(centroid[1] - 1, centroid[1] + 1)
    plt.show()
    print(80 * '#')
    print()
    return centroid, close_pixels, return_type


def get_common_pixels(pixel_list, point, return_indices=False):
    '''Find common pixels in /pixel_list/ that /point/ is a part of.'''
    inside = np.where([np.isin(point, pixel).all()
                      for pixel in pixel_list])[0]
    if len(inside) == 0:
        return None
    if return_indices:
        return inside
    return [pixel_list[i] for i in inside]


def get_full_common_pixel_cycle(pixel_list, point):
    '''Get cycle of common pixels which contains a node with /point/.'''
    common_inds = get_common_pixels(pixel_list, point, return_indices=True)
    total_inds = set(common_inds)
    added_any = True
    while added_any:
        added_any = False
        for ind in total_inds:
            for pixel in pixel_list[ind]:
                common_inds = get_common_pixels(pixel_list, pixel,
                                                return_indices=True)
                if not set(common_inds).issubset(total_inds):
                    added_any = True
                    total_inds = total_inds.union(set(common_inds))

    return total_inds


def decide_between_common_groups(pixel_list, point):
    '''Pick which common pixel in /pixel_list/ to leave /point/ in.'''
    # Take out any pixels that are in more than one of these groups
    total_pixel_cycle = list(get_full_common_pixel_cycle(pixel_list, point))
    cycle_pixels = [np.copy(pixel_list[i]) for i in total_pixel_cycle]
    for pixel in cycle_pixels:
        for pixel_point in pixel:
            if len(get_common_pixels(cycle_pixels, pixel_point)) > 1:
                # delete from all cycle pixels!
                for i, cycle_pixel in enumerate(cycle_pixels):
                    keep_inds = np.all(
                        ~np.isin(cycle_pixel, pixel_point), axis=1)
                    cycle_pixels[i] = cycle_pixel[keep_inds]
    # change empty list elements to inf to ignore
    for i, cycle_pixel in enumerate(cycle_pixels):
        if cycle_pixel.size == 0:
            cycle_pixels[i] = np.array([[np.inf, np.inf]])
    # get the centroids
    centroids = [calc_centroid(p) for p in cycle_pixels]
    dists_from_centroids = dists_from_point(centroids, point)
    return np.argmin(dists_from_centroids), centroids, total_pixel_cycle


def remove_instances(pixel_list, point, keep_inds, debug=0):
    '''Remove instaces of /point/ in /pixel_list/ that except at /keep_ind/.'''
    common_pixels = get_common_pixels(pixel_list, point, return_indices=True)
    pruned_pixel_list = copy(pixel_list)
    if common_pixels is None:
        raise Exception
    if debug:
        print("keep inds = ", keep_inds)
        print("common pixel = ", common_pixels)
    for ind in common_pixels:
        if ind not in keep_inds:
            if debug:
                print(f'deleting {ind}')
            pixel_keep_inds = np.all(~np.isin(pruned_pixel_list[ind], point),
                                     axis=1)
            pruned_pixel_list[ind] = pruned_pixel_list[ind][pixel_keep_inds]
    return pruned_pixel_list


def add_instances(pixel_list, point, add_inds, debug=0):
    '''Add instances of /point/ in /pixel_list/ at /add_inds/.'''
    add_pixel_list = copy(pixel_list)
    for ind in add_inds:
        if debug:
            print('add pixel = ', add_pixel_list[ind])
            print('point to add = ', point)
        assert not np.isin(add_pixel_list[ind], point).all()
        add_pixel_list[ind] = np.concatenate((add_pixel_list[ind], [point]))
    return add_pixel_list


def remove_emptys(pixel_list):
    '''Remove empty pixels from /pixel_list/'''
    emptys = np.where([len(x) == 0 for x in pixel_list])[0]
    while len(emptys) > 0:
        pop_val = pixel_list.pop(emptys[0])
        assert len(pop_val) == 0
        emptys = np.where([len(x) == 0 for x in pixel_list])[0]
    return pixel_list


def _merge_pixels(pixel_list, p1_ind, p2_ind, debug=0):
    '''Merge p1 and p2 within /pixel_list/ at /p1_ind/ and /p2_ind/.

    Assumed that p1 and p2 are a part of no other pixels.

    Does NOT remove empty elements in order to preserve list structure.'''
    for point in pixel_list[p1_ind]:
        pixel_list = remove_instances(pixel_list, point, [p2_ind], debug=debug)
        pixel_list = add_instances(pixel_list, point, [p2_ind], debug=debug)
    return pixel_list


def merge_pixels(pixel_list, merge_inds, debug=0):
    '''Merge each p1 and p2 within /merge_inds/ which index /pixel_list/.

    Assumes that each p1 and p2 are a part of no other pixels.

    Removes empty elements after the merging is completed.
    '''
    for (p1_ind, p2_ind) in merge_inds:
        pixel_list = _merge_pixels(pixel_list, p1_ind, p2_ind, debug=debug)
    return remove_emptys(pixel_list)


def clean_repeat_pixels(pixel_list, repeat_pixels, debug=0):
    '''Choose final pixels for duplicates in our /pixel_list/.'''
    for point in repeat_pixels:
        choice, _, cycle_pixels = decide_between_common_groups(
            pixel_list, point)
        keep_ind = cycle_pixels[choice]
        pixel_list = remove_instances(
            pixel_list, point, [keep_ind], debug=debug)
    # remove any zero length lists
    return remove_emptys(pixel_list)


def get_best_one_of_choice(pixel_list, point, centroid_dist=.4):
    '''See if /point/ should move to another pixel in /pixel_list/.

    If no pixel is good to move to, return None. Otherwise return that pixel.
    '''
    centroids = np.array([calc_centroid(p) for p in pixel_list])
    # ignore empty values
    centroids[np.isnan(centroids)] = np.inf
    dists_from_centroids = dists_from_point(centroids, point)
    # ignore current point
    dists_from_centroids[dists_from_centroids == 0] = np.inf
    if centroid_dist < np.min(dists_from_centroids):
        return None
    return np.argmin(dists_from_centroids)


def assign_one_of_pixels(pixel_list, debug=0):
    '''If appropriate move one-of pixels in /pixel_list/ to other groups.'''
    one_of_pixels = np.where([len(pixel) == 1 for pixel in pixel_list])[0]
    added_pixel_list = copy(pixel_list)
    for pixel in one_of_pixels:
        point = added_pixel_list[pixel][0]
        best_choice = get_best_one_of_choice(
            added_pixel_list, point)
        if debug:
            print(f'best_choice: {best_choice}')
        if best_choice is not None:
            added_pixel_list = remove_instances(
                added_pixel_list, point, [], debug=debug)
            added_pixel_list = add_instances(added_pixel_list, point,
                                             [best_choice], debug=debug)

    # prune off empty values at the end
    added_pixel_list = remove_emptys(added_pixel_list)
    return added_pixel_list


def clean_pixel_list(pixel_list, unused_points, verbose=0):
    '''
    1. Add any unused points to the nearest pixel if satisfying constraints
    2. Take care of any duplicate pixels
    3. Check one-ofs
    4. Check for any pixels that have >3 of each frequency type
    '''
    pixel_list = copy(pixel_list)

    # Add unused points to pixel_list
    for point in unused_points:
        pixel_list.append(np.array([point]))

    # Take care of duplicates
    unique, counts = np.unique(np.vstack(pixel_list), axis=0,
                               return_counts=True)
    dups = unique[counts > 1]
    if verbose:
        print(f"Number of repeats found = {len(dups)}. "
              "Choosing single pixels for each.")
    pruned_repeats = clean_repeat_pixels(pixel_list, dups, debug=0)

    # Check one-ofs
    one_of_pixels = np.where([len(pixel) == 1 for pixel in pruned_repeats])[0]
    if verbose:
        print(f"Number of one-ofs found = {len(one_of_pixels)}. "
              "Checking to possibly assign each to other pixels.")
    pruned_one_ofs = assign_one_of_pixels(pruned_repeats, debug=0)

    one_of_pixels = np.where([len(pixel) == 1 for pixel in pruned_one_ofs])[0]
    if verbose:
        print(f"Final number of one-ofs = {len(one_of_pixels)}.")
    return pruned_one_ofs


def check_frequency_types(pixel_list, pixel_dataframe, verbose=0):
    '''Check pixels in /pixel_list/ that have >2 of a frequency type.'''
    num_wrong = 0
    for i, pixel in enumerate(pixel_list):
        num_220s = 0
        num_280s = 0
        for point in pixel:
            df_value = pixel_dataframe[((pixel_dataframe['x_pos'] == point[
                0]) & (pixel_dataframe['y_pos'] == point[1]))]
            # if df_value.shape[0] != 1:
            #     print(df_value)
            # assert df_value.shape[0] == 1
            if df_value['msk_220'].values[0]:
                num_220s += 1
            else:
                assert df_value['msk_280'].values[0]
                num_280s += 1
        if num_220s > 2 or num_280s > 2:
            num_wrong += 1
            if verbose > 1:
                print(i, pixel, num_220s, num_280s, "has >2 frequency types")
            # if debug:
            #     get_pixel_debug(pixel_dataframe[['x_pos', 'y_pos']].values,
            #                     point)
    return num_wrong


def get_det_map(pixel_dataframe, verbose=1, **get_pixel_kwargs):
    '''Get final det map given a dataframe which contains x and y pos info.

    It should also have msk_220 and msk_280 info if versbose > 0. Any arguments
    in /get_pixel_kwargs/ are passed on to the 'get_pixel' function.
    '''
    num_already_in = 0
    already_in_list = []
    pixel_list = []
    interesting_returns = []
    for i, pos in enumerate(pixel_dataframe[['x_pos', 'y_pos']].values):
        if list(pos) in list(already_in_list):
            num_already_in += 1
            continue
        _, close_pixels, return_type = get_pixel(pixel_dataframe[[
            'x_pos', 'y_pos']].values, pos, debug=0, verbose=verbose,
            **get_pixel_kwargs)
        if return_type is not None and return_type > 0:
            interesting_returns.append(pos)

        if close_pixels is None or len(close_pixels) > 4:
            if verbose > 1:
                print(i, return_type, len(close_pixels))
            # _ = get_pixel_debug(
            #     pixel_dataframe[['x_pos', 'y_pos']].values, pos, debug=1)
            if close_pixels is None:
                print("algorithm didn't work on close pixels. Continuing.")
                continue
        pixel_list.append(close_pixels)
        already_in_list.extend(close_pixels.tolist())

    if verbose:
        print(f"Number of interesting return "
              f"values = {len(interesting_returns)}")

    unused_points = []
    for pos in pixel_dataframe[['x_pos', 'y_pos']].values:
        if list(pos) not in list(already_in_list):
            unused_points.append(pos)

    cleaned_pixel_list = clean_pixel_list(pixel_list, unused_points,
                                          verbose=verbose)

    if verbose:
        # Check for >2 of each frequency type
        num_wrong = check_frequency_types(cleaned_pixel_list, pixel_dataframe)
        print("Number of pixels with >2 of one frequency type = "
              f"{num_wrong}")
        return cleaned_pixel_list, pixel_list, interesting_returns

    return cleaned_pixel_list


def add_det_map_to_df(det_map, pixel_dataframe):
    '''Add pixels and centroids from /det_map/ to /pixel_dataframe/.'''
    for i, pixel in enumerate(det_map):
        for point in pixel:
            df_ind = (pixel_dataframe[['x_pos', 'y_pos']] == point).values.all(
                axis=1)
            pixel_dataframe.loc[df_ind, "pixel"] = i
            pixel_dataframe.loc[df_ind, "pixel_centroid_x"] = calc_centroid(
                pixel)[0]
            pixel_dataframe.loc[df_ind, "pixel_centroid_y"] = calc_centroid(
                pixel)[1]
