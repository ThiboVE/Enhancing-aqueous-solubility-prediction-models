#!/bin/bash

function get_result_url() {
    curl -s -H "x-api-key: ${APIKEY}" -X POST -d "$1" "$URL"/get_result_url
}

export -f get_result_url

BATCH_SIZE=1000
TOTAL=$(jq 'length' _to_download.json)
DOWNLOAD_DIR=../data/QuantumFP/QFP_output

downloaded_ids=$(ls "$DOWNLOAD_DIR" | sed -E 's/^internship_smiles2_([0-9]+)\..*/\1/')

# Update already downloaded molecules file
jq --argjson allowed "$(cat allowed_ids.json)" '[.[] | select(.Id as $id | ($allowed | index($id)))]' finished_batches.json > _to_download.json

jq 'length' _to_download.json

echo $TOTAL

for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
    echo "Processing items $i to $((i+BATCH_SIZE-1))"

    jq -c ".[$i:$((i+BATCH_SIZE))][]" _to_download.json \
    | parallel -j 50 get_result_url '{}' | jq -r '.DownloadURL' \
    | parallel -j 50 curl '{}' -OJ

done

# ----------------------------------------------------

function get_result_url() {
    curl -s -H "x-api-key: ${APIKEY}" -X POST -d "$1" "$URL"/get_result_url
}

export -f get_result_url

jq -c ".[]" _to_download.json \
    | parallel -j 50 get_result_url '{}' | jq -r '.DownloadURL' \
    | parallel -j 50 curl '{}' -OJ