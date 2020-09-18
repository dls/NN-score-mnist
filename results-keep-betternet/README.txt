Runs the system in KEEP mode, with the model enlarged.

$ diff ../2_train_and_run.jl 2_train_and_run.jl
114,115c114,116
<  	  Dense(prod(imgsize), 32, relu),
<     Dense(32, nclasses),
---
>  	  Dense(prod(imgsize), 256, relu),
>  	  Dense(256, 128, relu),
>     Dense(128, nclasses),
