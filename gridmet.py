import os
import requests
import pandas as pd
import xarray as xr
from tqdm import tqdm

class Gridmet:
    """
    A class to extract and process variables from NetCDF files for specific points
    or regions defined by shapefiles.
    """

    def __init__(self, data_path):
        """
        Initialize the Gridmet with the base data path.

        Parameters
        ----------
        data_path : str
            Directory where NetCDF files are stored.
        """
        self.data_path = data_path

        self.var_dict = {
            "pr": 'precipitation_amount', 
            "pet": 'potential_evapotranspiration', 
            "tmmx": 'air_temperature', 
            "tmmn": 'air_temperature',
            "vs": "wind_speed",
            "srad": "surface_downwelling_shortwave_flux_in_air",
            "rmax": "relative_humidity",
            "rmin": "relative_humidity"
        }
        
        self.colname_dict = {
            "pr": 'prec', 
            "pet": 'pet', 
            "tmmx": 'tmax', 
            "tmmn": 'tmin',
            "vs": "ws",
            "srad": "srad",
            "rmax": "rmax",
            "rmin": "rmin"
        }

        self.unit_dict = {
            "pr": 'mm', 
            "pet": 'mm', 
            "tmmx": 'K', 
            "tmmn": 'K',
            "vs": "m/s",
            "srad": "W/m2",
            "rmax": "%",
            "rmin": "%"
        }
        
        self.var_list = ["pr", "pet", "tmmx", "tmmn", "srad"]
    
    def download_nc_files(self, start_year, end_year, var_list=None):
        """
        Download NetCDF files for specified variables and years if not already present.

        Parameters
        ----------
        start_year : int
            Start year of the range.
        end_year : int
            End year of the range.

        Examples
        --------
        >>> extractor = Gridmet(data_path="/data/netcdf")
        >>> extractor.download_nc_files(out_path="/data/downloads", start_year=1980, end_year=2025)
        """
        if var_list is None:
            var_list = self.var_list
            
        for yr in tqdm(range(start_year, end_year + 1)):
            for var in var_list:
                filename = f"{var}_{yr}.nc"
                file_path = os.path.join(self.data_path, filename)

                if os.path.exists(file_path):
                    print(f"{filename} already exists, skipping download.")
                    continue

                print(f"Downloading {filename}")
                url = f"https://www.northwestknowledge.net/metdata/data/{filename}"
                response = requests.get(url, allow_redirects=True)

                if response.status_code == 200:
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
                else:
                    print(f"Failed to download {filename}: HTTP {response.status_code}")
    
    def extract_variable_for_point_over_years(self, var, lat, lon, start_year, end_year, file_template="{var}_{year}.nc"):
        """
        Extract time series data for a specific variable at a given point across multiple years
        from NetCDF files (WGS84,EPSG:4326).

        Parameters
        ----------
        var : str
            Short name of the variable (e.g., "pr", "tmmx").
        lat : float
            Latitude of the point. WGS84,EPSG:4326
        lon : float
            Longitude of the point. WGS84,EPSG:4326
        start_year : int
            Start year of the range.
        end_year : int
            End year of the range.
        file_template : str, optional
            Template for the NetCDF file names, default is "{var}_{year}.nc".

        Returns
        -------
        pd.DataFrame
            Combined time series data for the specified variable and years.

        Examples
        --------
        >>> extractor = Gridmet(data_path="/data/netcdf")
        >>> df = extractor.extract_variable_for_point_over_years(
        ...     var="pr",
        ...     lat=40.7128,
        ...     lon=-74.0060,
        ...     start_year=2000,
        ...     end_year=2020
        ... )
        >>> print(df.head())
        """
        combined_df = pd.DataFrame()
        var_long = self.var_dict[var]

        for year in range(start_year, end_year + 1):
            nc_file = os.path.join(self.data_path, file_template.format(var=var, year=year))

            if not os.path.exists(nc_file):
                print(f"File not found: {nc_file}. Skipping year {year}.")
                continue

            ds = xr.open_dataset(nc_file)
            variable_of_interest = ds[var_long]
            data_at_coord = variable_of_interest.sel(lat=lat, lon=lon, method="nearest")
            time_series_df = data_at_coord.to_dataframe()[[var_long]]
            time_series_df = time_series_df.rename_axis('Date')

            combined_df = pd.concat([combined_df, time_series_df])
        
        combined_df.columns = [self.colname_dict[var]]
        print(f"Complete extraction of {var} ({self.unit_dict[var]}) from {start_year} to {end_year}")
        return combined_df

    def extract_variable_for_shapefile_over_years(self, var, shapefile, start_year, end_year, operation="mean", file_template="{var}_{year}.nc"):
        """
        Extract time series data for a specific variable overlapping with a shapefile across multiple years
        from NetCDF files (WGS84,EPSG:4326), applying an aggregation operation.

        Parameters
        ----------
        var : str
            Short name of the variable (e.g., "pr", "tmmx").
        shapefile : geopandas.GeoDataFrame
            Shapefile containing the region of interest.
        start_year : int
            Start year of the range.
        end_year : int
            End year of the range.
        operation : str, optional
            Aggregation operation to apply to grid values. Options: "mean", "sum", None. Default is "mean".
        file_template : str, optional
            Template for the NetCDF file names, default is "{var}_{year}.nc".

        Returns
        -------
        pd.DataFrame
            Combined time series data for the specified variable and years.

        Examples
        --------
        >>> gdf = gpd.read_file("/data/shapefile/region.shp")
        >>> extractor = Gridmet(data_path="/data/netcdf")
        >>> df = extractor.extract_variable_for_shapefile_over_years(
        ...     var="pr",
        ...     shapefile=gdf,
        ...     start_year=2000,
        ...     end_year=2020
        ... )
        >>> print(df.head())
        """
        combined_df = pd.DataFrame()
        var_long = self.var_dict[var]
        
        for year in tqdm(range(start_year, end_year + 1)):
            nc_file = os.path.join(self.data_path, file_template.format(var=var, year=year))

            if not os.path.exists(nc_file):
                print(f"File not found: {nc_file}. Skipping year {year}.")
                continue

            ds = xr.open_dataset(nc_file)
            variable_of_interest = ds[var_long]

            shapefile = shapefile.to_crs(ds.rio.crs)
            variable_of_interest = variable_of_interest.rio.write_crs(ds.rio.crs)

            clipped_data = variable_of_interest.rio.clip(shapefile.geometry, shapefile.crs, drop=True)

            if operation == "mean":
                aggregated_data = clipped_data.mean(dim=["lon", "lat"])
            elif operation == "sum":
                aggregated_data = clipped_data.sum(dim=["lon", "lat"])
            else:
                raise ValueError("Invalid operation. Choose from 'mean' or 'sum'.")

            time_series_df = aggregated_data.to_dataframe()[[var_long]]
            time_series_df = time_series_df.rename_axis('Date')

            combined_df = pd.concat([combined_df, time_series_df])
            
        combined_df.columns = [self.colname_dict[var]]
        print(f"Complete extraction of {var} ({self.unit_dict[var]}) from {start_year} to {end_year}")
        return combined_df


#%%
r"""
# Initialize the object
data_path = "D:/Data/gridmet"
gridmet = Gridmet(data_path)

# Download nc files
gridmet.download_nc_files(1979, 2024)

# Example usage
start_year = 1980
end_year = 1985

# Extract point
lat = 41.8672778
lon = -75.2137500
df = gridmet.extract_variable_for_point_over_years("pr", lat, lon, start_year, end_year)

# Extract polygon
import geopandas as gpd
shapefile = gpd.read_file(r"C:\Users\CL\Documents\GitHub\YakimaRiverBasin\gis\yrb_data\yrb_umtw_catchment.shp")
df = gridmet.extract_variable_for_shapefile_over_years("pr", shapefile, start_year, end_year, operation="mean", file_template="{var}_{year}.nc")
"""