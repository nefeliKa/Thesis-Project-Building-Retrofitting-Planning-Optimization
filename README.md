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
1. First time during setup: `cd report && make docker`
2. In a windows command prompt: `wsl`
3. Make your changes...
4. `make simple` to get the `thesis.pdf` file