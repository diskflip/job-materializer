#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import yaml
import pandas as pd
from openai import OpenAI
from jobspy import scrape_jobs
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

CONFIG_DIR = Path("./config")
RESUME_PATH = CONFIG_DIR / "resume.txt"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
STATE_FILE = ".radar_state.json"

console = Console()

@dataclass
class Job:
    jid: str
    title: str
    link: str
    published: str
    published_dt: Optional[object]
    summary: str
    source: str
    company: str
    location: str


def load_resume() -> str:
    if not RESUME_PATH.exists():
        console.print(f"[yellow]No resume found at {RESUME_PATH}[/yellow]")
        return ""
    return RESUME_PATH.read_text(encoding="utf-8")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        console.print(f"[yellow]No config found at {CONFIG_PATH}[/yellow]")
        return {"min_score": 0}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"min_score": 0}


def init_client(api_key: str) -> Optional[OpenAI]:
    if not api_key:
        return None
    try:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    except Exception as e:
        console.print(f"[red]Failed to initialize OpenRouter:[/red] {e}")
        return None


def score_job_with_ai(client: OpenAI, job: Job, resume: str, config: dict) -> dict:
    goals = config.get("goals", "")
    background = config.get("background", "")
    pay = config.get("pay", "")
    location = config.get("location", "")
    evaluation_factors = config.get("evaluation_factors", "")
    current_time = datetime.now().strftime("%A %I:%M %p")
    
    prompt = f"""Score this job 0.0-1.0.

SCORING:
0.0-0.2: Skip
0.2-0.4: Stretch
0.4-0.6: Solid
0.6-0.8: Strong
0.8-1.0: RARE

CURRENT TIME: {current_time}
Late night/weekend posts by big US companies = ghost jobs.

RESUME:
{resume}

GOALS: {goals}
SITUATION: {background}
PAY: {pay}
LOCATION: {location}
FACTORS: {evaluation_factors}

JOB:
{job.title} at {job.company}
Location: {job.location}
{job.summary[:3000] if job.summary else 'No description'}

Return JSON:
- score: 0.0-1.0
- reasoning: LENGTH DEPENDS ON SCORE:
  * Under 0.4: One short sentence max (e.g. "Senior role, needs 5+ years")
  * 0.4-0.6: Two short lines with +/- (e.g. "+ skill match\\n- exp gap")
  * 0.6-0.8: 2-3 lines with +/- 
  * 0.8+: 3-4 lines with +/-
  Keep each line under 60 chars.
- should_apply: true/false"""

    try:
        response = client.chat.completions.create(
            model="x-ai/grok-4.1-fast",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        score = result.get("score", 0)
        if isinstance(score, float) and score <= 1.0:
            result["score"] = int(score * 100)
        result["score"] = max(0, min(100, result["score"]))
        return result
    except Exception as e:
        console.print(f"[dim]AI scoring failed: {e}[/dim]")
        return {"score": 0, "reasoning": "Scoring unavailable", "should_apply": False}


def stable_job_id(title: str, company: str) -> str:
    base = (title.strip().lower() + "||" + company.strip().lower()).encode("utf-8", errors="ignore")
    return hashlib.sha256(base).hexdigest()


def load_state(path: str) -> Dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"seen": [], "last_poll": {}}


def save_state(path: str, seen: List[str], last_poll: Dict, max_seen: int):
    if len(seen) > max_seen:
        seen = seen[-max_seen:]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"seen": seen, "last_poll": last_poll}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def safe_str(val) -> str:
    if val is None:
        return ""
    if pd.isna(val):
        return ""
    return str(val).strip()


def fetch_jobs_from_source(source: str, search_term: str, location: str, results_wanted: int, hours_old: int, proxy: str = None) -> List[Job]:
    jobs = []
    try:
        params = {
            "site_name": [source],
            "search_term": search_term,
            "location": location,
            "results_wanted": results_wanted,
            "hours_old": hours_old,
            "verbose": 0,
        }
        if proxy:
            params["proxies"] = proxy
        if source == "google":
            params["google_search_term"] = f"{search_term} jobs near {location} since yesterday"
        if source in ["indeed", "glassdoor"]:
            params["country_indeed"] = "USA"
        if source == "linkedin":
            params["linkedin_fetch_description"] = True
        
        df = scrape_jobs(**params)
        if df is None or df.empty:
            return jobs
        
        for _, row in df.iterrows():
            job_url = safe_str(row.get('job_url'))
            title = safe_str(row.get('title'))
            if not job_url or not title:
                continue
            
            date_posted = row.get('date_posted')
            published_dt = None
            published_str = ""
            if date_posted is not None and not pd.isna(date_posted):
                try:
                    if hasattr(date_posted, 'strftime'):
                        published_dt = date_posted
                        published_str = published_dt.strftime("%Y-%m-%d")
                    else:
                        published_str = str(date_posted)
                except Exception:
                    published_str = str(date_posted) if date_posted else ""
            
            city = safe_str(row.get('city'))
            state = safe_str(row.get('state'))
            loc_parts = [p for p in [city, state] if p]
            location_str = ", ".join(loc_parts) if loc_parts else "Unknown"
            company = safe_str(row.get('company')) or "Unknown"
            description = safe_str(row.get('description'))
            jid = stable_job_id(title, company)
            
            if any(j.jid == jid for j in jobs):
                continue
            
            jobs.append(Job(
                jid=jid,
                title=title,
                link=job_url,
                published=published_str,
                published_dt=published_dt,
                summary=description,
                source=source,
                company=company,
                location=location_str,
            ))
    except Exception as ex:
        console.print(f"[red]Error fetching from {source}:[/red] {ex}")
    return jobs


def get_site_name(url: str) -> str:
    if "indeed.com" in url:
        return "Indeed"
    if "linkedin.com" in url:
        return "LinkedIn"
    if "ziprecruiter.com" in url:
        return "ZipRecruiter"
    if "glassdoor.com" in url:
        return "Glassdoor"
    if "google.com" in url:
        return "Google"
    return "Link"


def render_job_card(job: Job, ai_reasoning: str = "", match_score: int = 0):
    from rich.box import ROUNDED
    
    if match_score >= 80:
        color = "bold magenta"
    elif match_score >= 60:
        color = "bold yellow"
    elif match_score >= 40:
        color = "blue"
    else:
        color = "white"
    
    company = job.company if job.company and job.company != 'Unknown' else "Unknown"
    location = job.location if job.location else "Unknown"
    site_name = get_site_name(job.link)
    
    max_width = min(76, console.width - 4)
    header_base = f"{company} | {job.title} | {location} | {site_name}"
    
    if len(header_base) > max_width:
        available = max_width - len(f"{company} |  | {location} | {site_name}") - 3
        title = job.title[:available] + "..." if available > 0 else job.title[:20] + "..."
    else:
        title = job.title
    
    body = Text()
    body.append(company, style=f"bold {color}")
    body.append(" | ", style="dim")
    body.append(title, style=f"bold {color}")
    body.append(" | ", style="dim")
    body.append(location, style=color)
    body.append(" | ", style="dim")
    body.append(site_name, style=f"underline {color} link {job.link}")
    
    if ai_reasoning:
        body.append("\n")
        for line in ai_reasoning.split("\n"):
            line = line.strip()
            if line.startswith("+"):
                body.append("+", style=f"bold {color}")
                body.append(line[1:] + "\n", style="white")
            elif line.startswith("-"):
                body.append("-", style=f"bold {color}")
                body.append(line[1:] + "\n", style="white")
            elif line:
                body.append(line + "\n", style="white")
    
    title_text = f"{match_score}%" if match_score > 0 else None
    panel = Panel(
        body,
        title=title_text,
        title_align="right",
        border_style=color,
        box=ROUNDED,
        padding=(0, 1),
        width=min(80, console.width),
    )
    console.print(panel)


def main():
    parser = argparse.ArgumentParser(description="Job Radar")
    parser.add_argument("--search", type=str, default="")
    parser.add_argument("--location", type=str, default="USA")
    parser.add_argument("--state", default=STATE_FILE)
    parser.add_argument("--max-seen", type=int, default=5000)
    parser.add_argument("--initial-limit", type=int, default=20)
    parser.add_argument("--hours-old", type=int, default=24)
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--indeed-interval", type=int, default=3)
    parser.add_argument("--zip-interval", type=int, default=5)
    parser.add_argument("--google-interval", type=int, default=8)
    parser.add_argument("--results", type=int, default=25)
    parser.add_argument("--indeed-only", action="store_true")
    parser.add_argument("--with-linkedin", action="store_true")
    parser.add_argument("--proxy", type=str, default=None)
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    
    search_term = args.search
    if args.dev and not search_term:
        search_term = "software engineer"

    resume = load_resume()
    config = load_config()
    min_score = config.get("min_score", 0)
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    client = None
    
    if not args.no_ai:
        if not api_key:
            console.print("[yellow]No API key - AI scoring disabled[/yellow]\n")
        elif not resume:
            console.print("[yellow]No resume - AI scoring disabled[/yellow]\n")
        else:
            client = init_client(api_key)
            if client:
                console.print("[green]AI scoring enabled[/green]\n")

    if args.indeed_only:
        sources = [{"name": "indeed", "interval": args.indeed_interval}]
    else:
        sources = [
            {"name": "indeed", "interval": args.indeed_interval},
            {"name": "zip_recruiter", "interval": args.zip_interval},
            {"name": "google", "interval": args.google_interval},
        ]
        if args.with_linkedin:
            sources.append({"name": "linkedin", "interval": 30})

    if args.reset_state and os.path.exists(args.state):
        try:
            os.remove(args.state)
        except Exception:
            pass

    state = load_state(args.state)
    seen_list = state.get("seen", [])
    seen = set(seen_list)
    last_poll = state.get("last_poll", {})
    first_run = len(seen_list) == 0

    console.print("[bold]Job Radar[/bold]")
    console.print(f"Search: {search_term or '(all jobs)'}")
    console.print(f"Location: {args.location}")
    if min_score > 0:
        console.print(f"Min score: {min_score}%")
    console.print("Sources: " + ", ".join(s["name"] for s in sources))
    console.print()
    if first_run:
        console.print("[dim]Loading recent jobs...[/dim]\n")
    console.print("[dim]Ctrl+C to stop[/dim]\n")

    try:
        while True:
            now = time.time()
            for source_cfg in sources:
                source_name = source_cfg["name"]
                interval = source_cfg["interval"]
                last = last_poll.get(source_name, 0)
                
                if now - last < interval and not first_run:
                    continue
                
                ts = datetime.now().strftime("%I:%M %p").lstrip("0")
                print(f"\r{' ' * 80}\r", end="")
                print(f"{ts} Polling {source_name}...", end="\r", flush=True)
                
                jobs = fetch_jobs_from_source(
                    source=source_name,
                    search_term=search_term,
                    location=args.location,
                    results_wanted=args.results,
                    hours_old=args.hours_old,
                    proxy=args.proxy,
                )
                last_poll[source_name] = time.time()
                
                if first_run:
                    jobs = jobs[:args.initial_limit]
                
                pending_jobs = [job for job in jobs if job.jid not in seen]
                total_pending = len(pending_jobs)
                
                if total_pending == 0:
                    ts = datetime.now().strftime("%I:%M %p").lstrip("0")
                    print(f"\r{' ' * 80}\r", end="")
                    print(f"{ts} {source_name}: no new jobs", end="\r", flush=True)
                
                for i, job in enumerate(pending_jobs):
                    seen.add(job.jid)
                    seen_list.append(job.jid)
                    
                    score = 0
                    reasoning = ""
                    
                    if client and resume:
                        remaining = total_pending - i
                        ts = datetime.now().strftime("%I:%M %p").lstrip("0")
                        print(f"\r{' ' * 80}\r", end="")
                        print(f"{ts} Scoring ({remaining} pending): {job.title[:40]}...", end="\r", flush=True)
                        
                        job_score = score_job_with_ai(client, job, resume, config)
                        score = job_score["score"]
                        reasoning = job_score["reasoning"]
                        print(f"\r{' ' * 80}\r", end="")
                        
                        if score < min_score:
                            continue
                    
                    render_job_card(job, ai_reasoning=reasoning, match_score=score)
                
                save_state(args.state, seen_list, last_poll, args.max_seen)
            
            if first_run:
                first_run = False
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped[/dim]")
        save_state(args.state, seen_list, last_poll, args.max_seen)


if __name__ == "__main__":
    main()
