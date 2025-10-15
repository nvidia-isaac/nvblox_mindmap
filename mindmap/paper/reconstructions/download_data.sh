#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR="$SCRIPT_DIR/input"

DATASET_NAME="mindmap_paper"

DOWNLOADED_DIR="$DATA_DIR/$DATASET_NAME"

# Add command line argument for upload flag
while getopts "uh" OPTION; do
    case $OPTION in
        u)
            UPLOAD=true
            ;;
        h)
            echo "Download the mindmap paper data"
            echo "Usage:"
            echo "download_data.sh -u"
            echo ""
            echo "  -u Upload (rather than download) the data to the server."
            echo "  -h help (this output)"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# Upload or download the data
if [ "$UPLOAD" = true ]; then
    osmo dataset upload $DATASET_NAME $DATA_DIR/*
else
    osmo dataset download $DATASET_NAME $DATA_DIR
    cp -r $DOWNLOADED_DIR/* $DATA_DIR
    rm -r $DOWNLOADED_DIR
fi
