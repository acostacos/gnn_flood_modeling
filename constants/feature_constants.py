from enum import StrEnum

class FeatureClass(StrEnum):
    GRAPH = "graph_features"
    NODE = "node_features"
    EDGE = "edge_features"

class FeatureType(StrEnum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    RAW = "raw"

class FeatureSource(StrEnum):
    HDF = "hdf"
    SHP ="shp"
