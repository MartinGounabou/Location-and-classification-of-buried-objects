# INFORMATIONS

## LIBs

Version des libs :

- Python 3.8.10
- Sklearn 0.24.2
- Pickle qui vient en général avec python

## Fonctionnement

Pour la prédiction, le modèle utilise les données des 13 dipôles. A partir de 10 mesures recueillies par chaque dipôle, soit 130 données, le modèle prédit l'altitude(z) à laquelle se trouve le squid.

Le modèle sera appelé chaque 10 mesures, avec en entrée les 13\*10 données dans l'ordre [dp12, dp13, dp16, dp18, dp26, dp27, dp28, dp36, dp37, dp38, dp45, dp58, dp68].

En sortie, l'algorithme nous donnera l'altitude à laquelle on se trouve.

## Types de données

- Entrée : numpy.array( , dtype = float64)
- Sortie : float64

## Autres infos utiles !

Il aura peut être des problèmes de compatibilité car le modèle a été developpé sur du 64 bits et sera tester apparamment sur du 32bits mais à voir ...


