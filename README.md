# MTL Street Sign to CurbLR

Transform Montreal parking sign data into CurbLR data format. For the CurbLR specification, take a tour of the [CurbLR Specification](https://github.com/curblr/curblr-spec).

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.6 or newer
- pip (Python package installer)
- Internet access for fetching the data from [Montreal Open Data](https://donnees.montreal.ca/dataset).
- [SharedStreets command line interface](https://github.com/sharedstreets/sharedstreets-js) for generating linear references for input data before converting it into the CurbLR format.

Note: The SharedStreets CLI currently runs on macOS and Linux. It does not (yet) support Windows.

## Getting Started

1. **Create a Virtual Environment**

   It is recommended to create a virtual environment to manage your project dependencies. You can create and activate a virtual environment as follows:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**

   Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the SharedStreets CLI**

   ```bash
   apt install npm
   npm install -g @sharedstreets/cli
   ```

## Usage

The script can be run with different options:

- `-p` or `--preprocessing` to enable preprocessing, which fetches and cleans the data.
- `-c` or `--curblr` to start the conversion.

**Note:** The preprocessing `-p` option needs to be run only once to generate the cleaned data locally. After that, just using the `-c` option is sufficient.

### Help

For help and a description of the options:

```bash
python3 main.py -h
```

## Example

Ensure you have an active internet connection to fetch the data from Montreal's Open Data portal. Here’s an example of how to run the script to preprocess the data and convert it to CurbLR:

```bash
python3 main.py -p -c
```

This command will first preprocess the data and then start the processing to convert the Montreal parking sign data into the CurbLR format.

[//]: # (## License)

[//]: # ()
[//]: # (This project is licensed under the ----- License—see the [LICENSE]&#40;LICENSE&#41; file for details.)