#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-12 上午11:10
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
import osmnx as ox
import networkx as nx
from utils import data_utils
from conf.configure import Configure
# remove warnings
import warnings

warnings.filterwarnings('ignore')


def main():
    if os.path.exists(Configure.processed_train_path.format('8')):
        return

    train, test = data_utils.load_dataset(op_scope='7')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    # Settings for Streetnetwork-Download
    STREETGRAPH_FILENAME = 'streetnetwork.graphml'
    FORCE_CREATE = False

    print 'download and load street area graph...'
    # This Checks if the Streetnetwork File exists (or creation is overwritten using FORCE_CREATE)
    if (not os.path.isfile('./data/'+STREETGRAPH_FILENAME)) or FORCE_CREATE:
        # There are many different ways to create the Network Graph. See the osmnx documentation for details
        area_graph = ox.graph_from_place('New York, USA', network_type='drive_service')
        ox.save_graphml(area_graph, filename=STREETGRAPH_FILENAME)
    else:
        area_graph = ox.load_graphml(STREETGRAPH_FILENAME)

    def driving_distance(raw):
        """
        Calculates the driving distance along an osmnx street network between two coordinate-points.
        The Driving distance is calculated from the closest nodes to the coordinate points.
        This can lead to problems if the coordinates fall outside the area encompassed by the network.

        Arguments:
        area_graph -- An osmnx street network
        startpoint -- The Starting point as coordinate Tuple
        endpoint -- The Ending point as coordinate Tuple
        """
        startpoint = (raw['pickup_latitude'], raw['pickup_longitude'])
        endpoint = (raw['dropoff_latitude'], raw['dropoff_longitude'])
        # Find nodes closest to the specified Coordinates
        node_start = ox.utils.get_nearest_node(area_graph, startpoint)
        node_stop = ox.utils.get_nearest_node(area_graph, endpoint)

        # Calculate the shortest network distance between the nodes via the edges "length" attribute
        distance = nx.shortest_path_length(area_graph, node_start, node_stop, weight="length")

        return distance

    print 'calc osmnx distance features...'
    conbined_data['osmnx_distance'] = conbined_data[['pickup_latitude','pickup_longitude',
                                                     'dropoff_latitude','dropoff_longitude']].apply(driving_distance, axis=1)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    train['trip_duration'] = trip_durations

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='8')


if __name__ == '__main__':
    print '========== generate osmnx distance features =========='
    main()
