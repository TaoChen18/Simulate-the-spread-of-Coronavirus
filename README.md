# Simulate-the-spread-of-Coronavirus
Our project is inspired by the Game of Life assignments and the spread of coronavirus. Basically, the universe is also a two-dimensional orthogonal grid of square cells (with no wrap around).

## Initialization
In the input file, we have an initial board data of 10 states and each state will have different initialization. For example, we create an initial 6*6 board of New York States as follows. The value of each cell can be 0 to 3. If a cell recovers or dies (with value 2 or 3), their value will never be changed in the future.  
**value 0: infected with the COVID-19 infection.  
value 1: not infected with the COVID-19 infection.  
value 2: recover from the COVID-19 infection  
value 3: die from COVID-19 infection **

Ex.
  New York: 
    000010 
    001010 
    000010 
    010010 
    000010 
    001010

## Spread Rules
1. Any live cell(value0)with an eighbor infected byCOVID-19 will have 12.5% to also infected by virus. (In other words, if all of its 8 neighbors are infected by virus, it will be 100% infected by virus.)
2. Any infected cell (value1) after 10 iterations will have 5% to die. Otherwise it will recover from the virus.

## MPI I/O Implementation
For each state, we assume they will only be adjacent to two other states and they need to share their borders and update their first and last columns based on their neighbor cell values. Our project will not have wrap around because it does not make sense for the geographic location to have wrap around. For our first state, we can assume the left neighbor column of its first column is all 0s. The same for the last column of the last state. However, since they are different states with different systems, they need to update their board in the huge global file and then read the neighbor data they need for each iteration.

Ex.
  MASSACHUSETTS:    NEW YORK:     PENNSYLVANIA:
    010010            000010          000010
    000011            000010          100011
    000010            001011          001010
    010011            000010          100010
    000010            000011          000011
    001010            000010          000010
