# Regles pour github : 

1. On créer des test pour chaque fonction que l'on écrit pour eviter des créer des bugs plus tard

2. On push les modifications sur une nouvelle branche
On fait une Pull Request pour merger la nouvelle branche avec main

3. Il faut l'autorisation de l'autre pour la PR

# Trame code : 
## Facilite le developpement
1. On verifie que la création de nos Unit Test fonctionnent comme l'on veut

2. On créer des fonctions de debug(afficher l'etat de n'importe quel class que l'on crée)

3. On créer notre notre API pour interagir avec d'autres bot avec le language standard

## La logique

4. On fait un arbre des débuts de partie grandmaster

5. On doit normaliser le reprsentation d'une piece
    * Un bitset pour ses positions ?

6. On doit stocker une position
    * Le tour
    * Positions des pieces (12 bitsets un pour chaque type)
    * Permissions de roque
    * Dernier coup joué (pour la prise en passant)
    * coder efficacement les règles de nulle/pat/échec et maths

7. On doit stocker un coup
    * Position de depart
    * Position de fin
    * Piece prise (pas necessaire si on charge une nouvelle situation a chaque fois)

8. On doit determiner les coups possible à jouer
    * Pseudo-legaux et puis filtrer les legaux


    * Les pieces qui attaquent le roi (Si il y en a)
        * Les pieces qu'on a le droit de prendre
        * Les cases où l'on a le droit de se positionner
    * Les cases où le roi peut se positioner
    * Les cases où les pions peuvent se positionner (+1, +2, diag, en passant)
    * Les cases où les cavaliers peuvent se positionner (On precalcule pour chaque position)
    * Les cases où les pieces glissantes peuvent se positionner (dame,tour,fou) precalcul+tric ?

9. On evalue une position
    1. en fonction des pièces
    2. en fonction de leur activité
    3. en fonction de leur placement
    4. en fonction de ce qu'elle attaque ?
    5. en fonction de si elle est cloué
    6. en fonction du nombre de coups qu'il faudrait pour la libérer (sans se faire manger ?) (définition de  libérer):

        1. pour attaquer une pièce adverse ?
        2. pour aller au centre ?
        3. pour défendre une pièce attaqué ?
    7. fait-elle une triple répétition ?
    8. en fonction de si la position est ouverte
    9. en fonction de la couleur où est la structure de pions pour les fous
    10. en fonction de si la pièce est fixé (doit absolument défendre un pion)
    11. en fonction de la structure de pion :

        1. un pion par colonne
        2. pion arriéré sur colonne ouverte
        3. pion isolé
        4. pion sans pion en face pour gêner la promotion
        5. si il est protégé/protégeable par un autre pion (un pion sur la colonee à côté ne peux pas forcément le protéger) (si il n'est pas sur une colonne ouverte, pas grave)
    12. pièce enfermé
    13. en finale, si les pions sont éloigné, alors le cavalier est moins bon qu'un fous
    14. en fonction des combinaisons de pièces en finales (dame-cavalier > dame-fous)
    15. en fonction de l'espace pris par la structure de pions (plus ils sont avancés ensemble, plus on a d'espace, moins bien c'est d'échanger des pièces avec l'adversaire qui a moins d'espace)
    16. utiliser le rapport en les scores plutôt que la différence (c'est mieux d'avoir un cavalier contre rien que tour-cavalier contre tour)

10. On determine la meilleure position
    1. On regarde le temps restant et on fait en fonction
    2. utiliser l'algo min-max avec alpha-bêta purning (facile)
    3. trier les coups en fonction de leur score immédiat (facile)
    4. faire une table de hachage pour se souvenir des positions déjà vus (moyen) (il faut optimiser un max)
    5. augmenter petit-à-petit la profondeur (alpha-bêta purning avec table de hachage permet de ne pas perdre trop de temps) (facile)
    6. réduire la profondeur de calcul pour les trops mauvais coups (moyen, doit être dûr à bien calibrer)
    7. continuer de faire tous les coups tant qu'il y a prise/échec
    8. faire une IA pour mieux scorer les positions (si on arrive à la faire apprendre d'elle-même, c'est encore mieux, mais on peux aussi se baser sur des databases de forts joueurs si le programme se situe en dessous)

# faire un bon comparateur de version
1. la faire jouer contre la dernière/première version un nombre suffisant de fois, en étant noir/blanc
2. les faire jouer sur des positions différentes de débuts qui sont a peu près égales selon stockfish, de début de partie, de milieu de partie, de fin de partie