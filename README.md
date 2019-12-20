# Combinatorical optimalization
This program implements Ant Colony Optimization algorithm for a Graph Traversing Problem.

Info
-----
The Graph has a specified set of vertices |V| = random(10, 150), edges |E|, and edge's weights |W|.
Algorithm has to visit each vertex and find path with the lowest cost S = sum(foreach weights_on_path). 
Cost is calculated on the basis of "if current weight is greater than next weight, then add current weight * 10"; 

Technical things
-----
Requires:
* numpy
* matplotlib

Usage
-----
```
python3 play.py 
# or 	
sudo chmod +x play.py  
# then
./play.py  
```

Authors
-----
* **Miłosz Chodkowski_PUT Poznan** [777moneymaker](https://github.com/777moneymaker)
