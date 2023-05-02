# Koch Snowflake

### Matt Tyler - 1049833

An interactive Koch snowflake written in Python and displayed with Matplotlib.


## The Approach

The basic premise of my approach was to go for speed. To achieve this, especially in Python, I chose to create the snowflake using matrix operations. This meant I could implement it mostly using numpy, where the underlying C implementation is fast, and I could then just continue repeating patterns by essentially doing memcopy of arrays, translating and mirroring them as required.

In an attempt to speed up apparent load times, I implemented a simple unbounded Cache (as I knew the upper limit was 13 orders which fits nicely enough in memory) by using the LRU cache and setting the maxsize to none. I then created a new thread at the start of the program that runs a simple helper function that will start processing from order 13, while the main thread will load with order 1 displayed.

This had a little effect, however I found that a large amount of my bottleneck in speed comes from plotting the points, not generating them. 

I attempted a few variations of having multiple buffers, however I quickly ran into the limitations of using a graph plotting package like matplotlib as the main way to display. Some hacky attempts to get it to work were attempted, such as having two matplotlib rendering engines at once and swapping which one was being drawn on screen, however as matplotlib isn't thread safe, this turned out to be a dead end. 



## Prerequisites

Python3 

Dependencies

```Bash
pip install numpy matplotlib
```

## How to run

```Bash
python3 Koch.py
```