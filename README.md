# Description
## Primary Goal
The primary goal for now is the create a model that can answer as follows. 
```
Q: write about linked list.
A:
A linked list is a linear data structure where elements, called nodes, are connected using pointers. Each node contains two parts:
Data – stores the actual value.
Next – a pointer/reference to the next node in the list.
[image] A->B

There are different kinds of linked lists. One is singly linked list. There are also doubly linked lists
[image] A<-> B
```

So, the point is our model will be able to answer as a sequence of texts and images. Note that we ARE NOT PLANNIGN to generate images. We just want to generate the connection like `NODE 1 -> NODE 2` or something like that.
