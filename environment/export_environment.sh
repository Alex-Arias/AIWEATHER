#!/bin/bash

echo "Exporting AIWeather environment..."

conda env export --no-builds > environment.yml
pip freeze > requirements.txt

echo "Done."