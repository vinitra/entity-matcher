## Project 1: Sigmod contest 2021
> Team members: Akash DHASADE, Angelika ROMANOU, Vinitra SWAMY, Eleni ZAPRIDOU  
Task: [Entity Matching](https://dbgroup.ing.unimo.it/sigmod21contest/task.shtml)

### Project Overview


### Methodology


#### Blocking


#### Clustering


### Experiments & Results

> [Leaderboard](https://dbgroup.ing.unimo.it/sigmod21contest/leaders.shtml)


### Code structure
The code is implemented in a modular and configurable fashion so as to decouple main logic with the orchestration and the hyper-parameter tuning. 

Main components of the application are:
- ```data_loader.p``` : Loads a dataset given its name. 
- ```blocking.py``` : Performs the blocking functionality as described in section X.
- ```clustering.py``` : Performs the clustering functionality as described in section X.
- ```evaluation.py``` : Compares predicted pairs with the actual ones and returns calssification scores (precision, recall, f1-score).
- ```pipeline.py``` : Performs the main logic of the app. It brings together the data with the blocking, the clustering and the evaluation. 
- ```tuning.py``` :  For a set of hyper-parameters for blocking and clustering method, it runs the pipeline and stores evaluation scores.
- ```main.py``` :  It runs the submitted pipeline, it creates and stores the output.csv file.

![code architecture](code-arch.png)
