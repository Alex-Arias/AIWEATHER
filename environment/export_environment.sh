#!/bin/bash

echo "Exporting Conda environment..."
conda env export --no-builds > environment/environment.yml

echo "Exporting pip requirements..."
pip freeze > environment/requirements.txt

echo "Done."