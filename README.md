## Development Setup
- Linux  
  - Nothing interesting here...


- Windows
  1. Install WSL
  2. Bring-up a (WSL) terminal (via the Terminal app)
  3. Setup your ssh keypair via `ssh-keygen`
  4. ```eval `ssh-agent -s` && ssh-add ~/.ssh/id_rsa ```
  

### Thesis code development
1. Clone this repo: `git clone git@github.com:jakeChal/nefeli-thesis.git`
2. Create a venv, activate it and install python packages:  
   `pip install -r requirements.txt`
3. Run:  
   E.g. `python retrofitting/run_episodes.py`
### Thesis writing
1. Clone https://github.com/cagix/pandoc-thesis
2. `make docker`
