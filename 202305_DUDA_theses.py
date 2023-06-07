#!/usr/bin/env python
# coding: utf-8

# # Analyse de theses.fr
# Ce travail a été accompli dans le cadre du Diplôme Universitaire "Data Analyst" (2023-24).
# Il s'appuie sur un jeu de données issu de thèses.fr (https://fil.abes.fr/2022/03/10/les-donnees-de-theses-fr-disponibles-sur-data-gouv-fr/). La version de travail est partagé dans le répertoire data.
# 
# Le travail a initialement été réalisé via Jupyter notebook.

# Avant de répondre aux différentes consignes, nous allons analyser et nettoyer le jeu de données.

# In[1]:


# On importe les librairies dont on aura besoin (Pandas, Matplotlib, Numpy...)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# on importe missingno - NB il a fallu passer par une installation via pip directement dans le notebook à la première utilisation
import missingno as msno

# On charge le fichier des thèses
data_theses=pd.read_csv("C:\DUDA\PhD_dataset.csv", header=0, low_memory=False)


# In[2]:


# On réalise un certain nombre de contrôle de surface

# Vérification du nombre de lignes et colonnes dans le DataFrame
print("\nStructure du fichier (.shape) : ")
print(data_theses.shape)

# Vérification des types de données et valeurs non nulles
print("\nInformation sur le jeu de données (.info)")
print(data_theses.info())

# Réalisation d'un summary du dataframe
print("\nDescription de la base : ",data_theses.describe())

# Vérification des valeurs manquantes pour chaque colonne
print("\nNombre de valeurs manquantes par colonne", data_theses.isnull().sum())

# Vérification manuelle des premières lignes du DataFrame
print("\nPremières lignes du jeu de donnnées : ")
print(data_theses.head(5))


# In[3]:


# On vide la mémoire pour charger un autre fichier.
import gc

del data_theses
gc.collect()


# In[4]:


# On charge le fichier PhD.v2
data_theses=pd.read_csv("C:\DUDA\PhD_v2.csv", header=0, low_memory=False)

# Vérification du nombre de lignes et colonnes dans le DataFrame
print("\nStructure du fichier (.shape) : ")
print(data_theses.shape)

# Vérification des types de données et valeurs non nulles
print("\nInformation sur le jeu de données (.info)")
print(data_theses.info())

# Réalisation d'un summary du dataframe
print("\nDescription de la base : ",data_theses.describe())

# Vérification des valeurs manquantes pour chaque colonne
print("\nNombre de valeurs manquantes par colonne", data_theses.isnull().sum())

# Vérification manuelle des premières lignes du DataFrame
print("\nPremières lignes du jeu de donnnées : ")
print(data_theses.head(5))


# In[5]:


# On vide à nouveau la mémoire et on charge la v3
del data_theses
gc.collect()
data_theses=pd.read_csv("C:\DUDA\PhD_v3.csv", header=0, low_memory=False)

# Vérification du nombre de lignes et colonnes dans le DataFrame
print("\nStructure du fichier (.shape) : ")
print(data_theses.shape)

# Vérification des types de données et valeurs non nulles
print("\nInformation sur le jeu de données (.info)")
print(data_theses.info())

# Réalisation d'un summary du dataframe
print("\nDescription de la base : ",data_theses.describe())

# Vérification des valeurs manquantes pour chaque colonne
print("\nNombre de valeurs manquantes par colonne", data_theses.isnull().sum())

# Vérification manuelle des premières lignes du DataFrame
print("\nPremières lignes du jeu de donnnées : ")
print(data_theses.head(5))


# # 1. Analyse des premiers contrôles de surface et nettoyage des données
# 
# 
# ## 1.1. Description du fichier et premiers contrôles
# Les contrôles de surface vont permettre d'identiier des actions de corrections de données. Parmi les problèmes classiques, on dénombre : 
# - le décalage de colonnes, lié à l'utilisation du séparateur dans les fichiers csv, 
# - les données manquantes,
# - l'encodage des caractères spéciaux.
# 
# Nous allons tenter d'identifier et proposer des solutions de redressement ou de contournement.
# 
# Le fichier contient 448047 entrées et 23 colonnes.
# 
# ## Les champs vides qui ne devraient pas l'être
# Comme il s'agit d'un fichier des thèses, il doit a minima avoir un auteur, un titre, un directeur de thèse. Globalement toutes les colonnes dont les valeurs non nulles sont au nombre de 448047 sont réputées complètes.
# Les informations de description indiquent néanmoins : 
# - il manque 7 titres de thèses ;
# - alors que les identifiants de directeur de thèse semblent complets, il manque 13 directeurs de thèse (nom, prénom).
# 
# Les autres écarts ne sont pas exploitables dans le cadre d'un contrôle de surface et indiquent plutôt des limites matérielles de collecte de données (la date de première inscription en doctorat est alimentée dans 14% des cas).
# 
# ## Autres remarques en première analyse du jeu de données
# La consultation des 10 premières lignes amène les remarques suivantes : 
# - il ne semble pas y avoir de caractères accentués problématiques (sauf dans le libellé des colonnes) ; 
# - l'identifiant auteur est connu dans 70% des cas ; 
# - la langue de la thèse semble peu souvent renseignée (on fait l'hypothèse que si la langue n'est pas renseignée, elle est écrite en Français) ;
# - le type de la colonne "Year" est float, cela devrait être un entier ; 
# - l'identifiant directeur semble être codé sur 8 caractères, et il y a des enregistrements qui dérogent à cette hypothèse ;  
# - la structure de l'identifiant these semble être du format s00000 (avec 4 ou 5 chiffres), certains enregistrements dérogent à cette hypothèse ; 
# - l'identifiant de l'établissement est connu dans 96% des cas - ce taux pourrait être amélioré en travaillant à partir du libellé de l'établissement qui est toujours présent (à une occurrence près).
# 
# On se propose d'explorer ces premiers jeux d'anomalies pour proposer quelques ajustements du jeu de données.

# In[6]:


# On sélectionne l'ensemble des lignes dont le titre de thèse est vide

# Sélection des lignes où le champ "Titre" est vide
titre_vide = data_theses[data_theses['Titre'].isnull()]

# Affichage du nombre de lignes où le champ "Titre" est vide
print("Nombre de lignes où le champ 'Titre' est vide :", len(titre_vide))

# Affichage de l'ensemble des colonnes pour les lignes où le champ "Titre" est vide
print(titre_vide)


# L'analyse des thèses sans titre permet de conclure qu'il s'agit d'une anomalie mais qu'il ne faut pas exclure les données puisque les thèses ont été soutenues. 

# In[7]:


# Sélection des lignes où le champ "Directeur de these" est vide
dir_these_vide = data_theses[data_theses['Directeur de these'].isnull()]

# Affichage du nombre de lignes où le champ "Directeur de these" est vide
print("Nombre de lignes où le champ 'Directeur de these' est vide :", len(dir_these_vide))

# Affichage de l'ensemble des colonnes pour les lignes où le champ "Directeur de these" est vide
print(dir_these_vide)


# Cette analyse ne permet pas d'écarter ces thèses qui ont été également soutenues. En revanche, le jeu de données écarte l'hypothèse formulée sur la structure du numéro identifiant de thèse. On pourrait imaginer que la structure a évolué au fil du temps. On pourra tester cette hypothèse si nécessaire, par la suite.

# In[8]:


# Pour identifier les problèmes éventuels liés aux caractères spéciaux, on recherche les cellules dont la valeur est #NAME?
count = np.count_nonzero(data_theses == "#NAME?")
print("Nombre de cellules égales à '#NAME?' dans data_theses : ", count)


# Le nombre de valeurs avec un problème d'encodage étant limité (21), on se propose d'écarter ses valeurs en cas de recherche de doublons, notamment.

# ## 1.2. Analyse des données manquantes
# Les données manquantes peuvent correspondre à quatre types de champs : vide, NaN, 0 alors qu'un nombre est attendu, ou tout autre anomalie (comme on l'a vu précédemment avec la valeur "#NAME?".
# N.B. : en analysant le jeu de données, on comprend que la langue peut être renseignée en 'na', la donnée est alors réputée présente mais ce n'est pas le cas.
# Pour cette première étape, on initialisera à 0 le maximum de valeurs (y compris pour les langues), alors que pour certains champs, on pourra prendre d'autres hypothèses de transformation.

# In[9]:


# On remplace maintenant l'ensemble des champs NaN par 0
# Comptage des champs vides
nb_nan = data_theses.isna().sum().sum()
print("Nombre de champs vides : ", nb_nan)

# Affichage des premières lignes du dataframe modifié
print(data_theses.head(15))


# In[10]:


# Calculer le pourcentage de valeurs manquantes
prc_missing = data_theses.isna().mean() * 100
# Création d'un DataFrame avec les pourcentages des valeurs manquantes
prc_missing_data = pd.DataFrame({"categories": prc_missing.index, "Pourcentage manquant": prc_missing.values})


# In[12]:


# Création du graphique
fig, ax = plt.subplots(figsize=(20,8))
msno.matrix(data_theses, ax=ax, sparkline=False)

# Modification des paramètres de l'axe des y
plt.yticks([])
plt.ylabel("")

# Modification des paramètres du titre
plt.title("Cartographie des données manquantes dans theses.fr", fontweight='bold', fontsize=18, y=-0.1)


# Affichage du graphique
plt.show()


# On enregistre le graphique en format PNG
plt.savefig("pourcentage_missing_data.jpg", dpi=300, bbox_inches='tight')


# In[13]:


# On recharge le fichier des thèses (le résultat précédent était incohérent)
data_theses=pd.read_csv("C:\DUDA\PhD_dataset.csv", header=0, low_memory=False)

# Corrélation des valeurs manquantes pour chaque colonne
msno.heatmap(data_theses)
plt.title("Heatmap des corrélations entre variables", fontweight='bold', fontsize=18)

plt.show()

# On enregistre le graphique en format PNG
plt.savefig("correlation_missing_data.jpg", dpi=300, bbox_inches='tight')


# In[14]:


# On crée une variable reprenant le pourcentage de données manquantes quand le statut de la thèse est "en cours"
missing_encours = pd.DataFrame(data_theses[data_theses['Statut'] == 'enCours'].isnull().sum()/len(data_theses[data_theses['Statut'] == 'enCours'])*100).reset_index()
missing_encours.columns = ['variable', 'pourcentage']
missing_encours = missing_encours.rename(columns={'pourcentage': 'pourcentage_enCours'})
print(missing_encours)

# On reproduit l'opération avec le statut "soutenue"
missing_soutenue = pd.DataFrame(data_theses[data_theses['Statut'] == 'soutenue'].isnull().sum()/len(data_theses[data_theses['Statut'] == 'soutenue'])*100).reset_index()
missing_soutenue.columns = ['variable', 'pourcentage']
missing_soutenue = missing_soutenue.rename(columns={'pourcentage': 'pourcentage_soutenue'})
print("\n",missing_soutenue)

# On fusionne les deux dataframes pour n'en faire qu'une seule
pourcent_manquant_soutenue_encours = pd.merge(missing_encours, missing_soutenue, on='variable', suffixes=('_enCours', '_soutenue'))
print("\n", pourcent_manquant_soutenue_encours)


# In[15]:


# On crée  dataframe avec les données pour la heatmap (je ne suis pas arrivé à une approche plus satisfaisante pour l'instant)
data = {'variable': ['Identifiant auteur', 'Titre', 'Directeur de these', 'Etablissement de soutenance', 'Identifiant établissement', 'Date première inscription', 'Date de soutenance', 'Year', 'Publication dans theses.fr', 'etablissement_rec', 'langue_rec'],
        '% pour les thèses en cours': [99.665602, 0.010497, 0.002999, 0.000000, 0.001500, 3.532922, 85.428944, 85.428944, 0.000000, 0.505346, 95.961732],
        '% pour les thèses soutenues': [16.751364, 0.000000, 0.002884, 0.000262, 4.478970, 100.000000, 0.030417, 0.030417, 0.046413, 0.717695, 0.033040]}

df = pd.DataFrame(data)

# On fait pivoter le dataframe
df_pivot = pd.melt(df, id_vars=['variable'], var_name='status', value_name='percentage')

# Créer la heatmap
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.heatmap(df_pivot.pivot("variable", "status", "percentage"), cmap="binary", annot=True, fmt=".2f", cbar=False, linewidths=.5)
plt.xticks(fontweight="bold", fontsize=9)
plt.yticks(fontweight="bold", fontsize=9)
plt.title("Données manquantes dans Thèses.fr", fontweight="bold", fontsize=12, pad=20)
plt.xlabel("")
plt.ylabel("")
plt.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=1.5)
plt.show()
plt.savefig('heatmap_theses.png', dpi=300, bbox_inches='tight')


# ## 1.3. Préparation du jeu de données (nettoyage)

# In[40]:


# On vide à nouveau la mémoire et on charge la v3
del data_theses
gc.collect()
data_theses=pd.read_csv("C:\DUDA\PhD_v3.csv", header=0, low_memory=False)


# In[41]:


# On renomme la première colonne en #No_enregistrement
data_theses.rename(columns={data_theses.columns[0]: "no_enregistrement"}, inplace=True)


# In[42]:


# Compter le nombre de valeurs différentes dans la colonne "Langue de la these"
nombre_valeurs_differentes = len(data_theses['Langue de la these'].unique())

# Afficher le résultat
print("Le nombre de valeurs différentes dans la colonne 'langue de la thèse' est de :", nombre_valeurs_differentes)

# Compter le nombre d'occurrences de chaque valeur unique dans la colonne "Langue de la thèse"
occurrences_langue_rec = data_theses['Langue de la these'].value_counts()

# Afficher le résultat
print("\n",occurrences_langue_rec)


# In[43]:


# On remplace la valeur 'na' par 'fr" dans la colonne 'langue de la thèse' ainsi que dans la colonne 'Langue_rec'
data_theses.loc[data_theses['Langue de la these'] == 'na', 'Langue de la these'] = 'fr'
data_theses['Langue_rec'].fillna('Français', inplace=True)


# In[44]:


# On identifie les autres valeurs non vides mais déclarées comme 'na'

# On identifie les auteurs non connus
nb_na = data_theses[data_theses['Auteur'] == 'na']['Auteur'].count()
print("Nombre de cellules avec le texte 'na' dans la colonne 'Auteur' : ", nb_na)

# On fait la même chose pour 'Identifiant directeur'
nb_na = data_theses[data_theses['Identifiant directeur'] == 'na']['Auteur'].count()
print("Nombre de cellules avec le texte 'na' dans la colonne 'Identifiant directeur' : ", nb_na)


# In[45]:


# On remplace les valeurs manquantes par 0
data_theses['Year'] = data_theses['Year'].fillna(0)

# On change le type de la colonne 'Year' pour qui soit entier 
data_theses['Year'] = data_theses['Year'].astype(int)
print(data_theses['Year'].head())


# In[46]:


# On identifie d'éventuelles valeurs aberrantes (inférieures à 1970 ou supérieures à 2023)
count_0 = data_theses['Year'].isna().sum() + (data_theses['Year'] == 0).sum()
count = ((data_theses['Year'] < 1970) | (data_theses['Year'] > 2023)).sum() - count_0
print("Nombre de valeurs aberrantes dans la colonne 'Year':", count)


# In[47]:


# On procède à une série de contrôle de surface pour s'assurer de la qualité des données 

# Compter le nombre de données à remplacer
n_missing = data_theses.isna().sum().sum()
print("Nombre de données à remplacer : ", n_missing)

# Remplacer les valeurs manquantes par zéro dans le DataFrame d'origine
data_theses.fillna(0, inplace=True)

# Vérifier les types de données
data_theses['Date de soutenance'] = pd.to_datetime(data_theses['Date de soutenance'])
data_theses['Year'] = pd.to_numeric(data_theses['Year'], errors='coerce')


# In[48]:


# Identifier les thèses en double sur la base du champ "Titre"
doublons_these = data_theses[data_theses.duplicated(subset='Titre', keep=False)]

# On identifie le nombre de thèse dont le titre est mal interprêté 
nb_titre_name = (data_theses['Titre'] == '#NAME?').sum()
nb_titre_zero = (data_theses['Titre'] == 0).sum()
nb_titre_point = (data_theses['Titre'] == '.').sum()
nb_titre_triplepoint = (data_theses['Titre'] == '...').sum()
nb_titre_vide = (data_theses['Titre'] == '').sum()

print("Nombre de thèses avec un titre du type \n - '#NAME?' :", nb_titre_name, "\n - 0 : ", nb_titre_zero, "\n - vide :", nb_titre_vide, "\n - '.' : ", nb_titre_point, "\n - '...':", nb_titre_triplepoint)

# Supprimer les thèses dont le titre est égal à 0, "", ".", ".." ou à #NAME? - ces valeurs ont été définies de façon itérative
# en fonction du jeu de valeurs
doublons_these = doublons_these.query('Titre not in ["", ".", "...", "#NAME?", 0]')

# On compte et on affiche le nombre de lignes concernées 
nb_doublons_these = len(doublons_these)
print("Nombre de thèses dont le titre se retrouve au moins deux fois dans la base : ", nb_doublons_these)

# Trier les thèses en doublon par titre après avoir sélectionné les colonnes principales
doublons_these = doublons_these[['Auteur', 'Titre', 'Identifiant auteur', 'Directeur de these']]
doublons_these = doublons_these.sort_values(by='Titre')

# Afficher les 16 premières lignes des thèses en doublon
print(doublons_these.head(16))


# Comme le nombre de thèse en doublon commence à être significatif (2.000 lignes sur 400.000), on se propose d'affiner l'identification de thèse en double sur la base d'un couple "auteur/titre" - on se propose de dédoublonner ensuite les thèses sur la base de cette clé, en classant par ordre d'identifiant auteur et en conservant la dernière valeur.

# In[49]:


# On peut affiner la définition du doublon en considérant comme doublon les cas où l'auteur et le titre sont identiques.
doublons_these_titre_auteur = data_theses[data_theses.duplicated(subset=['Auteur', 'Titre'], keep=False)]
doubons_these_titre_auteur = doublons_these_titre_auteur.sort_values(["Titre", "Auteur", "Identifiant auteur"])

print(len(doublons_these_titre_auteur))
print (doublons_these_titre_auteur.head(15))


# In[50]:


# On supprime les doublons en conservant la dernière ligne de chaque doublon
data_theses = data_theses.drop_duplicates(subset=["Titre", "Auteur", "Identifiant auteur"], keep="last")

# Comme il y a des doublons du fait d'écart de prénom, on procède à un autre dédoublonnage sur la base de la clé 'Titre'/'Directeur de these'
data_theses = data_theses.drop_duplicates(subset=["Titre", "Directeur de these"], keep="last")


# In[51]:


# Vérification des types de données et valeurs non nulles
print("Information sur le jeu de données (.info)")
print(data_theses.info())


# In[52]:


# On relance le code qui dénombre le nombre de lignes dont le titre se retrouve au moins deux fois dans le jeu de données
# pour observer l'impact des requêtes de dédoublonnage
# Identifier les thèses en double sur la base du champ "Titre"
doublons_these = data_theses[data_theses.duplicated(subset='Titre', keep=False)]

# Supprimer les thèses dont le titre est égal à 0, "", ".", ".." ou à #NAME? - ces valeurs ont été définies de façon itérative
# en fonction du jeu de valeurs
doublons_these = doublons_these.query('Titre not in ["", ".", "...", "#NAME?", 0]')

# On compte et on affiche le nombre de lignes concernées 
nb_doublons_these = len(doublons_these)
print("Nombre de thèses dont le titre se retrouve au moins deux fois dans la base : ", nb_doublons_these)

# Trier les thèses en doublon par titre après avoir sélectionné les colonnes principales
doublons_these = doublons_these[['Auteur', 'Titre', 'Identifiant auteur', 'Directeur de these']]
doublons_these = doublons_these.sort_values(by='Titre')

# Afficher les 16 premières lignes des thèses en doublon
print(doublons_these.head(16))


# Il reste 634 lignes de thèse avec le même titre. Il serait opportun d'affiner ce travail.

# In[53]:


# On filtre les thèses avec une date de soutenance non nulle
data_theses = data_theses[(data_theses['Date de soutenance'].notnull()) & (data_theses['Year'] >= 1975) & (data_theses['Year'] <= 2020)]

# On convertit la colonne "Date de soutenance" en datetime et on extrait l'année
data_theses["Year"] = pd.to_datetime(data_theses["Date de soutenance"]).dt.year

# On groupe par année et on compte le nombre de thèses
theses_par_an = data_theses.groupby("Year")["Titre"].count()

# On trace le graphique
plt.plot(theses_par_an.index, theses_par_an.values)
plt.title("Nombre de thèses soutenues par an", fontweight="bold", fontsize=12, pad=20)
plt.show()

# On enregistre le graphique en format PNG
plt.savefig("nombre_theses_soutenues_an.jpg", dpi=300, bbox_inches='tight')


# # 3. Détection d'un problème dans les données
# ## 3.1. Représentation de la distribution du mois de soutenance pour l’intégralité du jeu de données (1984-2018) 
# (Pourquoi le choix de s’arrêter en 2018 ?).
# Vous devez trouver les lignes de commandes pour que les mois soient ordonnés
# dans le bon ordre (janvier à gauche, décembre à droite). Il s’agit ici d’une simple
# distribution. Pour ce faire, vous devrez convertir la variable associée à la date
# dans un format qui pourra facilement être traité.

# In[54]:


# On crée un sous-ensemble avec les thèses soutenues entre 1984 et 2018
these_soutenue = data_theses[(data_theses["Date de soutenance"].notnull()) & (data_theses["Date de soutenance"] != "") & (data_theses["Year"].between(1984, 2018))]

# Affichage des premières lignes du sous-ensemble
print(these_soutenue.info())
print(these_soutenue.head())


# In[55]:


# On compte le nombre de thèses soutenues avec un statut "en cours" 
nb_encours = these_soutenue.loc[these_soutenue['Statut'] == 'enCours', 'Statut'].count()
print("Nombre de thèses soutenues au statut 'en cours' :", nb_encours)


# In[56]:


# On filtre les thèses avec une date de soutenance entre 1984 et 2018 inclus
these_soutenue = these_soutenue[(these_soutenue['Year'] >= 1984) & (these_soutenue['Year'] <= 2018)]

# On convertit la colonne 'Date de soutenance' en datetime avec le format '%d-%m-%y'
these_soutenue['Date de soutenance'] = pd.to_datetime(these_soutenue['Date de soutenance'], format='%d-%m-%y')

# On ajoute la colonne 'mois_soutenance' avec le format demandé
these_soutenue['mois_soutenance'] = these_soutenue['Date de soutenance'].apply(lambda x: f"{x.strftime('%m')}-{x.strftime('%b').lower()}")

print(these_soutenue.head())


# In[57]:


# On crée une distribution pour représenter le mois de soutenance de la thèse

# On compte le nombre de thèses par mois de soutenance
nb_these_mois = these_soutenue['mois_soutenance'].value_counts()

# On trie les mois par ordre chronologique
sorted_months = ['01-jan', '02-feb', '03-mar', '04-apr', '05-may', '06-jun', '07-jul', '08-aug', '09-sep', '10-oct', '11-nov', '12-dec']

# On crée l'histogramme avec Matplotlib
plt.figure(figsize=(12, 6))
plt.bar(sorted_months, [nb_these_mois[month] for month in sorted_months])

plt.title('Mois de soutenance des thèses entre 1984 et 2018', fontweight='bold', fontsize=12, pad=20) 
plt.show()


# In[58]:


# On calcule le pourcentage de thèses soutenues en janvier par année
pourcentages_janvier = these_soutenue.groupby(these_soutenue['Date de soutenance'].dt.year)['mois_soutenance'].apply(lambda x: (x == '01-jan').sum() / len(x) * 100)

# On crée le graphique
fig, ax = plt.subplots(figsize=(10,6))

ax.plot(pourcentages_janvier.index, pourcentages_janvier.values, color='darkblue', linewidth=2)
ax.set_title('Pourcentage de thèses soutenues en janvier par année (1984-2018)', fontweight='bold', fontsize=12, pad=20)
ax.yaxis.set_major_formatter('{x:.0f}%')

plt.show()
plt.savefig('pourcent_mois_theses.png', dpi=300, bbox_inches='tight')


# ## 3.2. Création d'un facetgrid pour présenter les distributions entre 2005 et 2018
# Représentez en premier lieu la distribution du mois de soutenance pour chaque année, de 2005 à 2018 dans un facetgrid. Pour réaliser ce travail, vous devrez suivre une logique de group.by sur deux variables (le mois et l’année),
# puis un count. 

# In[106]:


# Filtrage des données entre 2005 et 2018
these_filtrees = these_soutenue.loc[(these_soutenue["Year"] >= 2005) & (these_soutenue["Year"] <= 2018)]

# Comptage du nombre de thèses par mois pour chaque année
these_par_mois = these_filtrees.groupby(["Year", "mois_soutenance"]).count()["no_enregistrement"].reset_index()

# Définir l'ordre des mois pour l'affichage
mois_ordre = ['01-jan', '02-feb', '03-mar', '04-apr', '05-may', '06-jun', '07-jul', '08-aug', '09-sep', '10-oct', '11-nov', '12-dec']

# Création du facetgrid
g = sns.FacetGrid(data=these_par_mois, col="Year", col_wrap=4, sharey=False, col_order=range(2005, 2019))
g.map(sns.barplot, "mois_soutenance", "no_enregistrement", order=mois_ordre)
g.set_titles("{col_name}", fontweight='bold')
g.set(xticklabels=[])
g.set(xlabel='')
g.set(ylabel='')
g.set(ylim=(0, these_par_mois.no_enregistrement.max()))
g.fig.suptitle("Nombre de thèses soutenues par mois et par année", y=1.03, fontsize=12, fontweight='bold')

# Réglage de l'espace entre les subplots
plt.subplots_adjust(wspace=0.2, hspace=0.5)

# Taille du graphe
g.fig.set_size_inches(12.8, 9.6)

plt.show()
plt.savefig('facetgrid_mois_these_2005-2018.png', dpi=300, bbox_inches='tight')


# L'option la plus pertinente serait alors d'écarter les thèses réputées soutenues le 1er janvier.

# In[62]:


# On écarte les thèses soutenues le 1er janvier et on génère les nouveaux résultats.
# On ajoute la colonne 'soutenue_0101' avec la valeur 'O' si la thèse a été soutenue un premier janvier et 'N' dans le cas contraire
these_soutenue['soutenue_0101'] = these_soutenue['Date de soutenance'].apply(lambda x: 'O' if x.strftime('%d-%m') == '01-01' else 'N')
print(these_soutenue.head(5))


# In[63]:


# Nombre de thèses soutenues le 1 janvier
# Comptage des valeurs 'O' et 'N'
nb_soutenue_0101 = len(these_soutenue[these_soutenue['soutenue_0101'] == 'O'])

# Affichage des résultats
print(f"Nombre de thèses soutenues le 1er janvier : {nb_soutenue_0101} sur 378800 thèses soutenues.")


# In[108]:


# On régénère le facetgrid en écartant les thèses soutenues le 1er janvier

# Filtrage des données entre 2005 et 2018, non soutenues le 01/01
these_filtrees = these_soutenue.loc[(these_soutenue["Year"] >= 2005) 
                                    & (these_soutenue["Year"] <= 2018) 
                                    & (these_soutenue["soutenue_0101"] == "N")]

# Comptage du nombre de thèses par mois pour chaque année
these_par_mois = these_filtrees.groupby(["Year", "mois_soutenance"]).count()["no_enregistrement"].reset_index()

# Définir l'ordre des mois pour l'affichage
mois_ordre = ['01-jan', '02-feb', '03-mar', '04-apr', '05-may', '06-jun', '07-jul', '08-aug', '09-sep', '10-oct', '11-nov', '12-dec']


# Création du facetgrid
g = sns.FacetGrid(data=these_par_mois, col="Year", col_wrap=4, sharey=False, col_order=range(2005, 2019))
g.map(sns.barplot, "mois_soutenance", "no_enregistrement", order=mois_ordre)
g.set_titles("{col_name}", fontweight='bold')
g.set(xticklabels=[])
g.set(xlabel='')
g.set(ylabel='')
g.set(ylim=(0, these_par_mois.no_enregistrement.max()))
g.fig.suptitle("Nombre de thèses soutenues par mois et par année", y=1.03, fontsize=12, fontweight='bold')

# Réglage de l'espace entre les subplots
plt.subplots_adjust(wspace=0.2, hspace=0.5)

# Taille du graphe
g.fig.set_size_inches(12.8, 9.6)

plt.show()
plt.savefig('facetgrid_mois_these_2005-2018_sans0101.png', dpi=300, bbox_inches='tight')


# ## 3.3. On crée un graphe présentant le nombre moyen de thèses publiées par mois

# In[104]:



# Création d'un nouveau dataframe avec le nombre de thèses soutenues par mois et par année
these_filtrees_mois_annee = these_filtrees.groupby(['Year', 'mois_soutenance'])['no_enregistrement'].count().reset_index()
these_filtrees_mois_annee.rename(columns={'no_enregistrement': 'nombre_de_theses'}, inplace=True)

# Calcul de la moyenne et de l'écart-type des thèses par mois consolidée sur toutes les années
moyennes_et_ecarts_types_par_mois = these_filtrees_mois_annee.groupby('mois_soutenance')['nombre_de_theses'].agg(['mean', 'std']).reset_index()

# Définition des données pour l'histogramme
mois = moyennes_et_ecarts_types_par_mois['mois_soutenance']
moyennes = moyennes_et_ecarts_types_par_mois['mean']
ecarts_types = moyennes_et_ecarts_types_par_mois['std']

# Création de l'histogramme
fig, ax = plt.subplots(figsize=(10,6))

# Histogramme pour la moyenne
ax.bar(mois, moyennes, color='#3274A1', align='center', capsize=10)

# Trait vertical pour l'écart-type
ax.errorbar(mois, moyennes, yerr=ecarts_types, fmt='none', ecolor='black', elinewidth=1, capsize=5)

ax.set_title('Nombre moyen de thèses soutenues par mois entre 2005 et 2018', fontweight='bold', y=1.05)

plt.show()


# ## 3.3. Hyperdocteurs ou homonymes ? Le cas "Cécile Martin"

# In[34]:


# Création d'un masque booléen pour sélectionner les lignes dont l'auteur est "Cécile Martin" ou "Cecile Martin"
mask = (data_theses["Auteur"] == "Cécile Martin") | (data_theses["Auteur"] == "Cecile Martin")

# Application du masque pour sélectionner les lignes correspondantes
theses_c_martin = data_theses.loc[mask]

# Affichage du résultat
print(theses_c_martin.head(7))


# In[35]:


# On filtre les thèses de Cécile Martin sur les données les plus significatives
theses_c_martin.to_csv('theses_c_martin.csv', index=False)


# In[36]:


# Déceler les autres "hyperdocteurs" => cette partie est inachevée.

# On compte le nombre de thèses soutenues par auteur
nb_theses_par_auteur = these_soutenue.groupby(["Titre", "Identifiant auteur"])["Identifiant auteur"].count()

# On compte le nombre de personnes ayant soutenu au moins 3 thèses
nb_personnes_plus_de_3_theses = (nb_theses_par_auteur >= 3).sum()

print("Nombre de personnes ayant soutenu au moins 3 thèses :", nb_personnes_plus_de_3_theses)
Le code commence par créer un sous-ensemble these_soutenue contenant les thèses 


# # 4. Détection des outliers
# 

# In[37]:


# On crée un sous-ensemble avec les thèses soutenues entre 1984 et 2018
these_soutenue = data_theses[(data_theses["Date de soutenance"].notnull())
                             & (data_theses["Date de soutenance"] != "") 
                             & (data_theses["Year"].between(1984, 2018))]

# On compte le nombre de thèses encadrées par chaque directeur
nb_theses_encadrees = these_soutenue['Directeur de these'].value_counts()

# On crée un sous-ensemble avec les directeurs de thèse
directeurs_these = these_soutenue[['Directeur de these']].drop_duplicates()

# On écarte les directeurs de thèse inconnus
directeurs_these = directeurs_these[directeurs_these["Directeur de these"] != "Directeur de these inconnu"]

# On ajoute une colonne pour le nombre de thèses encadrées par chaque directeur
directeurs_these['Nb_theses_encadrees'] = directeurs_these['Directeur de these'].map(nb_theses_encadrees)
directeurs_these = directeurs_these.sort_values(by='Nb_theses_encadrees', ascending=False)

print(directeurs_these.head(15))


# # 5. La langue de soutenance

# In[62]:


# On calcule la proportion de chaque langue pour chaque année de soutenance
proportion_langue_par_annee = these_soutenue.groupby(['Year', 'Langue_rec']).size().unstack()
proportion_langue_par_annee = proportion_langue_par_annee.apply(lambda x: x/x.sum(), axis=1)

# On change l'ordre d'affichage des langues
proportion_langue_par_annee = proportion_langue_par_annee[['Français', 'Bilingue', 'Anglais', 'Autre']]

# On trace le graphe
ax = proportion_langue_par_annee.plot(kind="area", stacked=True)
ax.set_title("Proportion des langues des thèses soutenues par année", fontweight='bold', y=1)

# On change l'affichage de l'axe des y et des x
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals], fontweight='bold')

ax.set_xlabel('')


# On enlève le cadre et on met une grille blanche sur fond blanc
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(color='white')

# On positionne la légende en haut à droite à l'extérieur du graphique
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


# On affiche le graphe
plt.show()
plt.savefig('nb_theses_mois.png', dpi=300, bbox_inches='tight')


# In[ ]:




