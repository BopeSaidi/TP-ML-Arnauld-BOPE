# TP-ML-Arnauld-BOPE
## Régression & Classification

Ce dépôt GitHub regroupe deux travaux pratiques (TP) dédiés à la mise en œuvre d'algorithmes fondamentaux de machine learning. 
Les projets ont été développés en Python, avec l'environnement Anaconda et Jupyter Notebook pour une expérience reproductible et interactive.

## 📁 Contenu du dépôt

### TP 1 – Classification : Prédiction des clients potentiels

- **Objectif** : l'objectif de ce travail est d’identificatifier des prospects ayant la plus forte probabilité de conversion en clients payants, afin d’optimiser l’allocation des ressources commerciales. 
- **Données** : 
### Description de données

Les données contiennent les différents attributs des prospects et les détails de leur interaction avec ExtraaLearn. 
Les données détaillées dans le notebook.   
- **Techniques utilisées** :
  - Nettoyage des données
  - Exploration de données : Analyse univarible et Bivaribles
  - Visualisation (matplotlib, seaborn)
  - Arbres de décision et forêts aléatoires
  - Séparation train/test
  - Matrice de confusion, c
  - lassification report

    📄 Fichier : `TP 1 Classification - Prediction_potentiel_clients.ipynb`

---

### TP 2 – Régression Linéaire : Prediction Chiffres d'affaire du supermarché SuperKart

- **Objectif** : l'objectif principal de ce travail, est d'analyser les données de vente du SuperKart qui est une entreprise qui possède une chaîne de supermarchés et de magasins d'alimentation proposant une large gamme de produits. afin de prédire le chiffre d'affaires futur de ses différents points de vente, d'établir une stratégie de vente dans les différentes villes et de planifier ses stocks en conséquence.
Les données détaillées dans le notebook.

- **Techniques utilisées** :
  - Nettoyage des données
  - Exploration de données : Analyse univarible et Bivaribles
  - Visualisation (matplotlib, seaborn)
  - Régression linéaire (Scikit-learn)
  - Reseau de Neuronne SNN
  - Évaluation des performances (MSE, R², etc.)

  📄 Fichier : `TP 2 Regression - Prediction_Chiffres_d'affaire.ipynb`

---

## 🛠️ Environnement requis

- Python 3.8+
- Jupyter Notebook
- Bibliothèques :
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

🔧 Installez les dépendances :

```bash
name: lab_ista_env
channels:
  - conda-forge
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
  - https://repo.anaconda.com/pkgs/msys2
dependencies:
  - anaconda_prompt=1.1.0=haa95532_0
  - anyio=4.6.2=py310haa95532_0
  - argon2-cffi=21.3.0=pyhd3eb1b0_0
  - argon2-cffi-bindings=21.2.0=py310h827c3e9_1
  - asttokens=2.0.5=pyhd3eb1b0_0
  - async-lru=2.0.4=py310haa95532_0
  - attrs=24.3.0=py310haa95532_0
  - babel=2.16.0=py310haa95532_0
  - beautifulsoup4=4.12.3=py310haa95532_0
  - bleach=6.2.0=py310haa95532_0
  - brotli-python=1.0.9=py310h5da7b33_9
  - bzip2=1.0.8=h2bbff1b_6
  - ca-certificates=2025.4.26=h4c7d964_0
  - certifi=2025.4.26=pyhd8ed1ab_0
  - cffi=1.17.1=py310h827c3e9_1
  - colorama=0.4.6=py310haa95532_0
  - comm=0.2.1=py310haa95532_0
  - debugpy=1.8.11=py310h5da7b33_0
  - decorator=5.1.1=pyhd3eb1b0_0
  - defusedxml=0.7.1=pyhd3eb1b0_0
  - exceptiongroup=1.2.0=py310haa95532_0
  - executing=0.8.3=pyhd3eb1b0_0
  - h11=0.14.0=py310haa95532_0
  - httpcore=1.0.2=py310haa95532_0
  - httpx=0.27.0=py310haa95532_0
  - intel-openmp=2024.2.1=h57928b3_1083
  - ipykernel=6.29.5=py310haa95532_1
  - ipython=8.30.0=py310haa95532_0
  - jedi=0.19.2=py310haa95532_0
  - jinja2=3.1.6=py310haa95532_0
  - json5=0.9.25=py310haa95532_0
  - jsonschema=4.23.0=py310haa95532_0
  - jsonschema-specifications=2023.7.1=py310haa95532_0
  - jupyter-lsp=2.2.0=py310haa95532_0
  - jupyter_client=8.6.3=py310haa95532_0
  - jupyter_core=5.7.2=py310haa95532_0
  - jupyter_events=0.12.0=py310haa95532_0
  - jupyter_server=2.15.0=py310haa95532_0
  - jupyter_server_terminals=0.4.4=py310haa95532_1
  - jupyterlab=4.3.4=py310haa95532_0
  - jupyterlab_pygments=0.3.0=py310haa95532_0
  - jupyterlab_server=2.27.3=py310haa95532_0
  - libblas=3.9.0=31_h641d27c_mkl
  - libcblas=3.9.0=31_h5e41251_mkl
  - libffi=3.4.4=hd77b12b_1
  - libhwloc=2.11.2=default_ha69328c_1001
  - libiconv=1.18=h135ad9c_1
  - liblapack=3.9.0=31_h1aa476e_mkl
  - libsodium=1.0.18=h62dcd97_0
  - libwinpthread=12.0.0.r4.gg4f2fc60ca=h57928b3_9
  - libxml2=2.13.8=h866ff63_0
  - markupsafe=3.0.2=py310h827c3e9_0
  - matplotlib-inline=0.1.6=py310haa95532_0
  - mistune=3.1.2=py310haa95532_0
  - mkl=2024.2.2=h66d3029_15
  - nbclient=0.10.2=py310haa95532_0
  - nbconvert-core=7.16.6=py310haa95532_0
  - nbformat=5.10.4=py310haa95532_0
  - nest-asyncio=1.6.0=py310haa95532_0
  - notebook-shim=0.2.3=py310haa95532_0
  - openssl=3.5.0=ha4e3fda_1
  - overrides=7.4.0=py310haa95532_0
  - packaging=24.2=py310haa95532_0
  - pandas=2.2.3=py310hb4db72f_3
  - pandocfilters=1.5.0=pyhd3eb1b0_0
  - parso=0.8.4=py310haa95532_0
  - patsy=1.0.1=pyhd8ed1ab_1
  - pip=25.0=py310haa95532_0
  - platformdirs=3.10.0=py310haa95532_0
  - prometheus_client=0.21.0=py310haa95532_0
  - prompt-toolkit=3.0.43=py310haa95532_0
  - prompt_toolkit=3.0.43=hd3eb1b0_0
  - psutil=5.9.0=py310h827c3e9_1
  - pure_eval=0.2.2=pyhd3eb1b0_0
  - pycparser=2.21=pyhd3eb1b0_0
  - pysocks=1.7.1=py310haa95532_0
  - python=3.10.16=h4607a30_1
  - python-dateutil=2.9.0post0=py310haa95532_2
  - python-fastjsonschema=2.20.0=py310haa95532_0
  - python-json-logger=3.2.1=py310haa95532_0
  - python-tzdata=2025.2=pyhd8ed1ab_0
  - python_abi=3.10=2_cp310
  - pywin32=308=py310h5da7b33_0
  - pywinpty=2.0.15=py310h72d21ff_0
  - pyyaml=6.0.2=py310h827c3e9_0
  - pyzmq=26.2.0=py310h5da7b33_0
  - referencing=0.30.2=py310haa95532_0
  - requests=2.32.3=py310haa95532_1
  - rfc3339-validator=0.1.4=py310haa95532_0
  - rfc3986-validator=0.1.1=py310haa95532_0
  - rpds-py=0.22.3=py310h636fa0f_0
  - scipy=1.15.2=py310h15c175c_0
  - send2trash=1.8.2=py310haa95532_1
  - setuptools=75.8.0=py310haa95532_0
  - sniffio=1.3.0=py310haa95532_0
  - soupsieve=2.5=py310haa95532_0
  - sqlite=3.45.3=h2bbff1b_0
  - stack_data=0.2.0=pyhd3eb1b0_0
  - statsmodels=0.14.4=py310hb0944cc_0
  - tbb=2021.13.0=h62715c5_1
  - terminado=0.17.1=py310haa95532_0
  - tinycss2=1.4.0=py310haa95532_0
  - tk=8.6.14=h0416ee5_0
  - tomli=2.0.1=py310haa95532_0
  - tornado=6.4.2=py310h827c3e9_0
  - traitlets=5.14.3=py310haa95532_0
  - typing-extensions=4.12.2=py310haa95532_0
  - typing_extensions=4.12.2=py310haa95532_0
  - ucrt=10.0.22621.0=h57928b3_1
  - urllib3=2.3.0=py310haa95532_0
  - vc=14.42=haa95532_4
  - vc14_runtime=14.42.34438=hfd919c2_26
  - vs2015_runtime=14.42.34438=h7142326_26
  - wcwidth=0.2.5=pyhd3eb1b0_0
  - webencodings=0.5.1=py310haa95532_1
  - websocket-client=1.8.0=py310haa95532_0
  - wheel=0.45.1=py310haa95532_0
  - win_inet_pton=1.1.0=py310haa95532_0
  - winpty=0.4.3=4
  - xz=5.6.4=h4754444_1
  - yaml=0.2.5=he774522_0
  - zeromq=4.3.5=hd77b12b_0
  - zlib=1.2.13=h8cc25b3_1
  - pip:
      - absl-py==2.1.0
      - astunparse==1.6.3
      - charset-normalizer==3.4.1
      - contourpy==1.3.1
      - cycler==0.12.1
      - flatbuffers==25.2.10
      - fonttools==4.56.0
      - gast==0.6.0
      - google-pasta==0.2.0
      - grpcio==1.71.0
      - h5py==3.13.0
      - idna==3.10
      - joblib==1.4.2
      - keras==3.9.0
      - kiwisolver==1.4.8
      - libclang==18.1.1
      - markdown==3.7
      - markdown-it-py==3.0.0
      - matplotlib==3.10.1
      - mdurl==0.1.2
      - ml-dtypes==0.5.1
      - namex==0.0.8
      - numpy==2.1.3
      - opt-einsum==3.4.0
      - optree==0.14.1
      - pillow==11.1.0
      - protobuf==5.29.3
      - pygments==2.19.1
      - pyparsing==3.2.1
      - pytz==2025.1
      - rich==13.9.4
      - scikit-learn==1.6.1
      - seaborn==0.13.2
      - six==1.17.0
      - tensorboard==2.19.0
      - tensorboard-data-server==0.7.2
      - tensorflow==2.19.0
      - tensorflow-io-gcs-filesystem==0.31.0
      - termcolor==2.5.0
      - threadpoolctl==3.6.0
      - tzdata==2025.1
      - werkzeug==3.1.3
      - wrapt==1.17.2
