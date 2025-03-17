# Projet de clustering des immages Digit pour ET4 info Polytech Paris Saclay

### step 1 : téléchargement des données et installation des packages
    - a. installer les requierements : "pip install -r requierements.txt"
### step 2 : configuration du chemin vers les donnés
    - dans le dossier src/constant.py, modifier la variable "PATH_DATA" par le chemin vers le dossier contenant les données à clusteriser.

### step 3 :  run de la pipeline clustering
    - aller dans le dossier src
    - exécutez la commande : "python pipeline.py"
    
### step 4 : lancement du dashboard
    - aller dans le dossier src 
    - exécutez la commande : "streamlit run dashboard_clustering.py"