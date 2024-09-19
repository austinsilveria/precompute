import json
import os
from os.path import expanduser

import pyarrow as pa
import pyarrow.parquet as pq

class ArtifactConstants:
    ARTIFACT_CACHE_PATH = f'{expanduser("~")}/precompute-artifact-cache/'

def write_artifact(metadata, columns) -> None:
    if not os.path.exists(ArtifactConstants.ARTIFACT_CACHE_PATH):
        print(f'making cache path: {ArtifactConstants.ARTIFACT_CACHE_PATH}')
        os.makedirs(ArtifactConstants.ARTIFACT_CACHE_PATH)

    metadata['metadata_path'] = f'{ArtifactConstants.ARTIFACT_CACHE_PATH}{metadata["name"]}-metadata.json'
    metadata['data_path'] = f'{ArtifactConstants.ARTIFACT_CACHE_PATH}{metadata["name"]}.arrow'

    with open(metadata['metadata_path'], 'w') as f:
        json.dump(metadata, f)
    table = pa.table(columns)
    pq.write_table(table, metadata['data_path'])

# List all artifacts in the artifact cache, sorted by time
def list_artifacts() -> list:
    artifacts = [ArtifactConstants.ARTIFACT_CACHE_PATH + f for f in os.listdir(ArtifactConstants.ARTIFACT_CACHE_PATH) if f.endswith('-metadata.json')]
    return sorted(artifacts, key=os.path.getmtime, reverse=True)

def read_artifact_metadata(file_path) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def read_artifact_data(file_path) -> pa.Table:
    return pq.read_table(file_path)