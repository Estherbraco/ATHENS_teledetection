"""Download Copernicus data from the Copernicus Data Space Ecosystem.

Uses the OpenEO API and a YAML query configuration.
"""

import os
import time

import openeo
import yaml
from openeo.rest import OpenEoApiError


def query_id(query, temporal_id=True):
    spatial_extent = ""
    for key, value in query["spatial_extent"].items():
        spatial_extent += key + "=" + str(value) + "-"
    temporal_extent = (
        query["temporal_extent"][0] + "-" + query["temporal_extent"][1]
    )
    if temporal_id:
        return os.path.join(
            query["collection"],
            spatial_extent[:-1],
            temporal_extent,
        )
    return os.path.join(query["collection"], spatial_extent[:-1])


if __name__ == "__main__":
    with open("cdse_query.yaml", "r") as file:
        query = yaml.safe_load(file)

    connection = openeo.connect(url="openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc()

    data_folder = os.path.join(query["data_folder"], query_id(query))

    for band in query["bands"]:
        cube = connection.load_collection(
            query["collection"],
            spatial_extent=query["spatial_extent"],
            temporal_extent=query["temporal_extent"],
            bands=[band],
            max_cloud_cover=query["max_cloud_cover"],
        )

        job = cube.save_result(
            format="GTiff",
            options={"filename_prefix": f"{query['collection']}_{band}"},
        ).create_job(title=f"S2 {band}")
        job.start_and_wait()
        target_folder = os.path.join(data_folder, band)
        os.makedirs(target_folder, exist_ok=True)

        for attempt in range(6):
            try:
                job.get_results().download_files(target_folder)
                break
            except OpenEoApiError as error:
                if "JobNotFinished" not in str(error) or attempt == 5:
                    raise
                print("Résultats pas encore disponibles, nouvelle tentative dans 10 secondes...")
                time.sleep(10)
