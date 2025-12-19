# Job Materializer - wip

A small agentic terminal tool that uses llm models to rank, display, and save potential job postings based on a mix of your resume and personal needs.

## Features

### Scrapes jobs from:

- Indeed

- ZipRecruiter

- Google Jobs

- LinkedIn 

## AI scoring

Saves high-scoring jobs to saved.txt (default threshold: 60% match) 

Persists state in .radar_state.json so you don’t see the same jobs repeatedly 

## Requirements

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

## Configuration

This script reads from ./config/:
- config/resume.txt (plain text resume)
- config/config.yaml (scoring preferences) 
- config/resume.txt

Put your resume in plain text. If it’s missing, the script still runs, but AI scoring will be disabled. 

#### config/config.yaml
#### Example:

min_score: 60

goals: >
  Backend / platform roles, strong engineering culture.

background: >
  5+ years Python, APIs, cloud, data pipelines.

pay: >
  Targeting $X+ base.

location: >
  Remote or Denver.

evaluation_factors: >
  Seniority fit, stack match, role scope, company quality.


min_score is in percent (0–100). If set, jobs scoring below this are skipped (when AI scoring is enabled). 


Set your API key:

export OPENROUTER_API_KEY=""


### Usage

#### Basic:

python radar.py --search "software engineer" --location "USA"


#### Run without AI scoring:

python radar.py --search "data engineer" --location "USA" --no-ai


#### Indeed-only mode:

python radar.py --search "backend engineer" --indeed-only


#### Include LinkedIn:

python radar.py --search "platform engineer" --with-linkedin
