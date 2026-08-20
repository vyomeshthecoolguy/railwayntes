@echo off
REM This launcher makes Task Scheduler setup simpler and more reliable.
REM It changes into the script's folder first, so raw_data.csv always
REM saves in the right place no matter how Task Scheduler invokes it.

cd /d "%~dp0"
python 08_scrape_ntes_client.py >> scrape_log.txt 2>&1
