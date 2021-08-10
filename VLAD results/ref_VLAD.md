Training parameters & Performance of other VLAD networks

|Name	|Input	|Loss	|Parameters	|Optimisation   |Metric	|Dataset & Performance
|:---   |:---   |:---   |:---   |:---   |:---   |:--- 
|NetVLAD    |RGB-Images	|Triplet loss	|K=64, all available positives, 10 hardest negatives and another 10 hardest from the previous epoch    |SGD, margin=0.1, lr=0.0001 or 0.001 and halved every 5 epochs, momentum=0.9, weight_decay=0.001, batch_size=4, trained for at most 30 epochs |25m, Top-N Recall	|Pittsburgh(Pitts250k), Tokyo 24/7
|PointNetVLAD   |point clouds   |Triplet / Lazy Triplet / Lazy Quadruplet   |K=64, 2 best/closest positive, 18 hardest/closest negatives   |alpha=0.5, beta=0.2, ...   |Top 1% Recall   |Oxford Robot Car, In-house datasets
|:---   |:---   |:---   |:---   |:---   |:---   |:--- 
|:---   |:---   |:---   |:---   |:---   |:---   |:--- 
|:---   |:---   |:---   |:---   |:---   |:---   |:--- 
