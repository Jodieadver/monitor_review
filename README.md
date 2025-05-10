# Monitor Review Analysis

This project analyzes monitor review data to extract insights about monitor features, pros, cons, and distributions across brands and segments.

## Features

- Loads and processes monitor review data
- Analyzes pros and cons distribution
- Extracts and analyzes key features from reviews
- Creates visualizations:
  - Pros vs Cons distribution
  - Brand distribution
  - Segmentation distribution
- Provides summary statistics

## Setup

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python monitor_analysis.py
```

## Output

The script will generate:
- Three visualization files (PNG format)
- Summary statistics in the console
- Analysis of top features mentioned in reviews

## Data Format

The script expects monitor review data with the following columns:
- model: Monitor model name
- brand: Manufacturer brand
- segmentation: Market segment (e.g., gaming, professional)
- pros: List of positive points (prefixed with (+))
- cons: List of negative points (prefixed with (-))
- text: Detailed review text
- country: Country where the review was written

## Extending the Analysis

To analyze your own monitor data:
1. Modify the `load_data()` function in `monitor_analysis.py`
2. Add your data in the same format as the example
3. Run the script to generate new analysis