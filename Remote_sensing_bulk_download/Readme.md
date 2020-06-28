# Bulk downloading Sentinel 3 satellite data using Jupyter Notebooks, SentinelSat python library
Included in this directory are example files for bulk downloading Sentinel 3 data, incuding an example JSON file for specifying a geographic region using a.json file.
## S3 products tested so far:
1. level 1 products (from Copernicus Scihub repository)
2. Level 2 SST products (from EUMETSAT Coda repository)
## General notes and observations
1. An account must first be created with the repository so that a correct username and password can be passed to the server when requesting data.
2. A JSON file can be used to pass a geolocation to Sentinelsat to specify a geographic region of interest. The contents of the JSON file can be created using the [geojson site](http://geojson.io/#map=2/20.0/0.0). An example JSON file corresponding to an area encompassing Malta and Sicily can be found in this directory.
3. Sentinel 3 has a very wide swath and hence rapid revisit time. Each Sentinel 3 dataset is ~500MB in size. Take care before executing “api.download_all(products)” that sufficient space is available in the local directory for the number of scenes that will be downloaded.
## Scihub notes and observations
1. From September 2019, ESA started archiving older data (acquired more than 12-18 months ago) into a Long Term Archive. This data is no longer available for immediate download and must instead be requested. Products restored from the long term archives are kept online for a period of at least 3 days. More information about the LTA is available [here](https://forum.step.esa.int/t/esa-copernicus-data-access-long-term-archive-and-its-drawbacks/15394).
2. From the Sentinelsat documentation:

>“Copernicus Open Access Hub no longer stores all products online for immediate retrieval. Offline products can be requested from the [LTA](https://scihub.copernicus.eu/userguide/LongTermArchive)  and should become available within 24 hours. Copernicus Open Access Hub's quota currently permits users to request an offline product every 30 minutes.”

Sentinelsat will throw up an error if you request a scene from the LTA twice in a 30 min period. This poses a problem for bulk downloading. It’s possible to time requests to the server to be sent every 30 mins. Presently, it seems there is an additional limit of 20 requests per day (this is to be confirmed).
## CODA notes and observations
1. EUMETSAT acknowledged in a private communication (sent April 2020) that there is a problem with specifying a geographic region to the CODA server such that daytime data (i.e. descending node) is not included for download when making queries to this server, unless the requested time period falls within the last 30 days prior to the date the request is made. Their suggestion is to specify the geolocation via the satellite orbit parameters. This is unsatisfactory and hopefully will be resolved soon. In addition, SST data from CODA is provided for the entire orbit, and not just the tile/region/row-path for the geo location that you specify.
2. While Scihub divides a given orbit into tiles, CODA provides data for the entire orbit in a single file.


