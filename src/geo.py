import fiona
import pyogrio
import geopandas as gpd


# Function to process layers from GDB file
def process_layers(gdb_path, layers):
    layer_dict = {}
    for name in layers:
        with fiona.open(gdb_path, layer=name) as layer:
            if layer.schema['geometry'] != 'None':
                gdf = gpd.read_file(gdb_path, layer=name)
                gdf["source_layer"] = name
                layer_dict[name] = gdf
            else:
                df = pyogrio.read_dataframe(gdb_path, layer=name)
                df["source_layer"] = name
                layer_dict[name] = df
    return layer_dict