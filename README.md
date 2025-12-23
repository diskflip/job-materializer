# Job Materializer

A small, agentic terminal tool that uses llms to rank, display, and save job postings based on a mix of your resume and personal needs.

## Features

### Scrapes jobs from:

- Indeed

- ZipRecruiter

- Google Jobs

- LinkedIn (soon)

## scoring

Saves high-scoring jobs to saved.txt

State saved to .radar_state.json

## Requirements
```
python -m venv .venv
source .venv/bin/activate  
```
```
pip install -r requirements.txt
```
## Configuration

This script reads from ./config/:
- config/resume.txt (plain text resume)
- config/config.yaml (scoring preferences) 

Put your resume in plain text. If it’s missing, the script still runs, but scoring will be disabled. 



### Set your API key:
```
export OPENROUTER_API_KEY=""
```

### Usage

```
python radar.py --search "software engineer" --location "USA"
```

#### Run without scoring:
```
python radar.py --no-ai
```

#### Indeed-only mode:
```
python radar.py --indeed-only
```

#### Include LinkedIn:
```
python radar.py --with-linkedin
```
