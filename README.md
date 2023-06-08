# DUDA_these
## Contexte : travail d'analyse des thèses (DU data analyst)
Ce travail a été réalisé dans le cadre d'un projet universitaire d'analyse des données de thèses afin de s'exercer à la manipulation des données (nettoyage, visualisation...). 

## Comment faire tourner le code
Le code a été écrit dans un notebook Jupyter. Il peut donc être lancé à partir de Jupyter Notebook.

## Installation des dépendances
Si la version de Jupyter est à jour (6.5.x ou supérieur), les librairies complémentaires sont importées dès les premières lignes. Elles sont énumérées dans le fichier requirements.txt (numpy, gc, pandas, matplotlib, seaborn, missingno).

## les données
Les données sont en open data, accessibles à l'adresse suivante : https://fil.abes.fr/2022/03/10/les-donnees-de-theses-fr-disponibles-sur-data-gouv-fr/.

Un extrait du jeu de données retravaillé (~1% de la base) est partagé sous la nom PhD_dataset.csv (le fichier complet pèse 153 Mo et contient 448047 enregistrements) - il est accessible sur demande, sa structure est la suivante : 

--   Column                                    Non-Null Count   Dtype  
---  ------                                    --------------   -----  
 0   Unnamed: 0                                448047 non-null  int64  
 
 1   Auteur                                    448047 non-null  object 
 
 2   Identifiant auteur                        317700 non-null  object 
 
 3   Titre                                     448040 non-null  object 
 
 4   Directeur de these                        448034 non-null  object 
 
 5   Directeur de these (nom prenom)           448034 non-null  object 
 
 6   Identifiant directeur                     448047 non-null  object 
 
 7   Etablissement de soutenance               448046 non-null  object 
 
 8   Identifiant etablissement                 430965 non-null  object 
 
 9   Discipline                                448047 non-null  object 
 
 10  Statut                                    448047 non-null  object 
 
 11  Date de premiere inscription en doctorat  64331 non-null   object 
 
 12  Date de soutenance                        390961 non-null  object 
 
 13  Year                                      390961 non-null  float64
 
 14  Langue de la these                        448047 non-null  object 
 
 15  Identifiant de la these                   448047 non-null  object 
 
 16  Accessible en ligne                       448047 non-null  object 
 
 17  Publication dans theses.fr                448047 non-null  object 
 
 18  Mise a jour dans theses.fr                447870 non-null  object 
 
 19  Discipline_prÃ©di                         448047 non-null  object 
 
 20  Genre                                     448047 non-null  object 
 
 21  etablissement_rec                         444973 non-null  object 
 
 22  Langue_rec                                383927 non-null  object 
