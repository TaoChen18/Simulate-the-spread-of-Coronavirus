# Simulate-the-spread-of-Coronavirus
Our project is inspired by the Game of Life assignments and the spread of coronavirus. Basically, the universe is also a two-dimensional orthogonal grid of square cells (with no wrap around).

## Initialization
In the input file, we have an initial board data of 10 states and each state will have different initialization. For example, we create an initial 6*6 board of New York States as follows. The value of each cell can be 0 to 3. If a cell recovers or dies (with value 2 or 3), their value will never be changed in the future.  
**value 0: infected with the COVID-19 infection.**  
**value 1: not infected with the COVID-19 infection.**  
**value 2: recover from the COVID-19 infection**  
**value 3: die from COVID-19 infection**  

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

## File nodes:
1. gol-main.c:  
includes all the MPI I/O function and the main function is defined in this file.  
First, it uses MPI read to read all the files and then process it in to the mode which can be easily executed in from the extern function(defined in gol-with-cuda.cu).Then used the the function of transferred rules of several iterations. This part  call the kernel function in cuda file and would be executed on GPU. It then executes the MPI write function. The total time has been recorded using ticks.

2. gol-with-cuda.cu:  
Defined all the function executed in cuda GPU which is actually the changing rules of every iteration.It starts with the memory allocation of GPU, then the initialize of all the parameter.  
The kernel function and related threads and blocks. The synchronization of parallel data. And at last swap the old and new world. These are all the  function  we would use in the main function.

3. sov-without-cuda.c:  
is no different from the gol-main.c. Except that the implementation of the changing world every iteration used the parallel mode in CPU not the parallel mode in GPU. Besides that, the MPI I/O is just the same and all the rules of changing is the same. So there is no extern function from gol-with-cuda.cu file.

4. slurmSpecturm.sh:  
is the shell file which is used to run the code after makefile compiled.   
**NOTICE**:  
The “scratch” in the last two lines should be replaced by the the path of the local execution file(after makefile).  
Every node only has 6 ranks and  the rank size needs to be the same as the state count.  
The 1st argument is the state amount/rank size  
The 2rd argument is the state height/width(it’s a square in our model)  
The 3rd argument is the thread count.  
The 4th  argument is the integration count.  
The 5th argument is the flag(1 indicates writing the result into file, otherwise no output in file.)  

5. makefile:  
includes all the commands needed to compile all the files and execute the program in the bash environment. Type in make in the bash all these commands would been executed.

6. g1.xlsx:  
Data for making graphs
