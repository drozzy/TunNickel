# Project Codename: Tun Nickel
Extension of the surgical data science course project.

# Setup
1. Download the JIGSAWS dataset and place the `Suturing` folder inside `src/tunnickel`. So that your folder structure looks as follows:

    ```
    TunNickel
    - src
      - tunnickel
        - Suturing
    ```

2. Install torchdyn 0.2.0: https://torchdyn.readthedocs.io/en/latest/index.html

3. Create conda env from provided environment.yml:

       conda env create -f environment.yml
       conda activate surgical

4. Run `experiment.py` from src dir:

       cd src
       python experiment.py -h
    
## Vscode setup guide
https://binx.io/blog/2020/03/05/setting-python-source-folders-vscode/
