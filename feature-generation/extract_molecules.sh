#!/bin/bash
function get_batch_status() {
    curl -s -H "x-api-key: ${APIKEY}" -X GET "$URL"/status/"$1"
}

get_batch_status internship_smiles2.json | jq -c 'map(select(.Status == "Finished")) | sort_by(.Id)' > finished_batches.json