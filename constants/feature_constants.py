from enum import StrEnum

class FeatureClass(StrEnum):
    NODE = "node_features"
    EDGE = "edge_features"

class FeatureType(StrEnum):
    STATIC = "static"
    DYNAMIC = "dynamic"

class FeatureSource(StrEnum):
    HDF = "hdf"
    SHP ="shp"
