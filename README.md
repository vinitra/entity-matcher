## Entity Matcher
> Contest: [SIGMOD Programming Contest 2021](https://dbgroup.ing.unimo.it/sigmod21contest)  
> Members: Akash DHASADE, Angelika ROMANOU, Vinitra SWAMY, Eleni ZAPRIDOU  
Task: [Entity Matching](https://dbgroup.ing.unimo.it/sigmod21contest/task.shtml)  
Team: sigmodest  

### Project Overview
This project refers to the SIGMOD programming contest 2021.
The task consists of identifying which instances, described by properties (i.e., attributes), represent the same real-world entity (entity resolution).
Participants are asked to solve the task among several datasets of different types (e.g., products, people, etc.). 
Each dataset is made of a list of instances (rows) and a list of properties describing them (columns).
The goal is to find, for each Xi dataset, all pairs of instances that match (i.e., refer to the same real-world entity).  
  
For more details, read the [official SIGMOD 2021 task description](https://dbgroup.ing.unimo.it/sigmod21contest/task.shtml).

### Methodology

![image](https://user-images.githubusercontent.com/72170466/116088668-5d96aa80-a6a2-11eb-9c39-cadfbd31fbf3.png)

In the following section, we describe our methodology. The implemented pipeline is comprised of two main parts; the blocking and the entity clustering. 
Blocking tries to minimize the number of comparisons by assigning rows to blocks from the input data based on a specific blocking key scheme.
Entity clustering examines the rows in each block and distills the entity-matching clusters.

#### Blocking
Blocking refers to the strategy of reducing number of potential comparisons between entities in the dataset using concept of blocks. A naive way to match entities would be to perform an `O(n^2)` comparison which is highly inefficient for very large datasets. Blocking generates groups or blocks of entities which are expected to be potential matches based on any simple criteria. Once the blocks are generated, entities are matched with other entities only within a block thereby significantly reducing number of comparisons. A very restrictive blocking scheme thus may hurt recall.

The entities in datasets X2 and X3 provided for the Sigmoid contest represent Laptops sold on different e-commerce websites. It is easy to infer that any two matching laptops will always have same `brand` and `cpu`. Thus `brand` and `cpu` formed our initial blocking key. The challenge here was to address missing values of these attributes for some of the tuples. To solve this problem we construct an exclusive set of all brands and cpus observed in the entire training dataset. The title attribute essentially contains data from all remaining columns as one large string (which possibly appears as description of item on e-commerce website). We then simply search for these brands and cpus (from the set) in the title attribute of the row with missing values. Below picture shows sample matching entities in the dataset. Observe the brand and cpu column, hypens separate matching entities.

<img width="408" alt="pic1" src="https://user-images.githubusercontent.com/24961068/116411503-12ad9c00-a836-11eb-88a2-894d9f3b89f8.png">

With this blocking key and further steps in the pipeline, we could only manage to achieve an F1 score of 50% on the test dataset. Thus we decided to further improve our blocking scheme.

#### Clustering
In this step, entity matching is implemented, where for each block collection and for each pair of candidate matches that co-occur in a block, it is decided if they refer to the same entity.
In our implementation we tested two types of approaches (five methods); two **distance similarity methods** and three **unsupervised learning methods**.  

We computed the similarity metrics based on the matched entities that correspond to the blocking key, namely:
- *Jaccard similarity* of each row to its corresponding blocking key group
- *Cosine similarity* of the encoded titles of each row to its corresponding blocking key group
  
We also distill the blocking key groups into smaller matching clusters using the following clustering algorithms
   - *K-Means Clustering*
   - *Agglomerative Clustering*
   - *Birch Clustering*
 
We also experimented with Mean-Shift clustering and using a BERT based sentence encoder, but found both inferior to the other clustering and encoding approaches. 

### Experiments & Results
We tested all the aforementioned approaches in all the provided datasets. The results for the best model of each type are presented in the following table.

| Model                  | Precision | Recall | F1-score |
|------------------------|:---------:|:------:|:--------:|
| Jaccard similarity (threshold: 0.75)  | 0.945      | 0.397   | 0.559     |
| K-means clustering (k: 2) with USE embeddings  | 0.881      | 0.505   | 0.642     |
| Birch clustering with USE embeddings  | 0.872      | 0.582   | 0.698     |
| Cosine similarity (threshold: 0.75) with USE embeddings  | 0.913      | 0.668   | 0.772     |
| Agglomerative Clustering (dist = 2) with USE embeddings  | 0.875      | 0.885   | **0.88**     |

 
After tuning our hyper-parameters, our optimal pipeline involves:
- Encoder: **Universal Sentence Encoder** (USE)
- Clustering method: **Agglomerative clustering** (distance threshold = 2)
- F1 score on leaderboard: **0.744**
- On 14.04.2021, our team **sigmodest** was 5th place in the rankings.

> [Leaderboard](https://dbgroup.ing.unimo.it/sigmod21contest/leaders.shtml)


### Code structure
The code is implemented in a modular and configurable fashion so as to decouple main logic with the orchestration and the hyper-parameter tuning. 

Main components of the application are:
- ```data_loader.py``` : Loads a dataset given its name. 
- ```blocking.py``` : Performs the blocking functionality as described in section [Blocking](#blocking).
- ```clustering.py``` : Performs the clustering functionality as described in section [Clustering](#clustering).
- ```evaluation.py``` : Compares predicted pairs with the actual ones and returns classification scores (precision, recall, f1-score).
- ```pipeline.py``` : Performs the main logic of the app. It brings together the data with the blocking, the clustering and the evaluation. 
- ```tuning.py``` :  For a set of hyper-parameters for blocking and clustering method, it runs the pipeline and stores evaluation scores.
- ```main.py``` :  It runs the submitted pipeline, it creates and stores the output.csv file.

![code architecture](code-arch.png)

### Deployment

Please follow the instruction below to run the entity matching pipeline.

Create `virtualenv` and install requirements.
```bash
pip install virtualenv
virtualenv <name_of_virtualenv>
source <name_of_virtualenv>/bin/activate
pip install -r requirements.txt
```

Run tuning (for all encoding + similarity + clustering combinations and resulting evaluations)
```bash
python src/tuning.py
```

Run main (for the optimal encoding + similarity + clustering combination from the leaderboard results).
```bash
python src/main.py
```
