# orb_profitability_assessment

## Create below directories strctures to store raw data.

```
|-data
    |-raw-data
        |-raw-data-1minute
            |-nse
                |-equity
                    |-2021
                        |-January_2021
                            |-.CNX100.csv
                        |-.....
                        |-December_2021
                    |-2022
                        |-January_2022
                        |-.....
                        |-December_2021

```

## Install required libraries from "requirements.txt"
    pip install --upgrade pip
    pip install -r requirements. txt    

## Ensure that the "data" folder follows appropriate folder structure & is included as a part of .gitignore file
    download data at https://drive.google.com/drive/folders/1pjOkfxDE1zY9lpzkSW6ZLIcG57rkvYml`
    
## Run the main function 
    cd src
    python orb_main.py

## Run the sensitivity analysis
    cs src
    python sensitivity_analysis.py